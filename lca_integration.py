#!/usr/bin/env python3
"""
Life-Cycle Assessment (LCA) Data Integration
Generates comprehensive LCA data for building materials, systems, and operations
for the Building DNA Dataset
"""

import numpy as np
import pandas as pd
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

@dataclass
class LCAMaterial:
    """LCA data for building materials"""
    material_id: str
    name: str
    category: str  # structural, insulation, finishing, etc.
    embodied_carbon: float  # kg CO2/kg
    embodied_energy: float  # MJ/kg
    water_consumption: float  # L/kg
    waste_generation: float  # kg/kg
    recyclability: float  # 0-1
    renewable_content: float  # 0-1
    transportation_distance: float  # km
    transportation_mode: str  # truck, rail, ship, air
    manufacturing_location: str
    disposal_method: str  # landfill, incineration, recycling, reuse

@dataclass
class LCAProcess:
    """LCA data for building processes"""
    process_id: str
    name: str
    category: str  # construction, operation, maintenance, demolition
    carbon_emissions: float  # kg CO2/m²/year
    energy_consumption: float  # MJ/m²/year
    water_consumption: float  # L/m²/year
    waste_generation: float  # kg/m²/year
    duration: float  # years
    frequency: float  # times per year
    impact_factors: Dict[str, float]  # GWP, AP, EP, ODP, POCP, etc.

@dataclass
class LCABuilding:
    """Complete LCA assessment for a building"""
    building_id: str
    assessment_date: str
    assessment_period: int  # years
    total_embodied_carbon: float  # kg CO2
    total_operational_carbon: float  # kg CO2/year
    total_embodied_energy: float  # MJ
    total_operational_energy: float  # MJ/year
    total_water_consumption: float  # L/year
    total_waste_generation: float  # kg/year
    carbon_intensity: float  # kg CO2/m²/year
    energy_intensity: float  # MJ/m²/year
    water_intensity: float  # L/m²/year
    waste_intensity: float  # kg/m²/year
    materials_breakdown: Dict[str, float]
    processes_breakdown: Dict[str, float]
    impact_categories: Dict[str, float]
    recommendations: List[str]

class LCAIntegration:
    """Life-Cycle Assessment data integration and generation"""
    
    def __init__(self, seed: int = 42):
        """Initialize the LCA integration"""
        np.random.seed(seed)
        random.seed(seed)
        self.materials_lca = self._initialize_materials_lca()
        self.processes_lca = self._initialize_processes_lca()
        self.impact_categories = self._initialize_impact_categories()
        
    def _initialize_materials_lca(self) -> Dict[str, LCAMaterial]:
        """Initialize LCA data for building materials"""
        materials = {
            'concrete': LCAMaterial(
                material_id='concrete',
                name='Concrete',
                category='structural',
                embodied_carbon=0.12,  # kg CO2/kg
                embodied_energy=0.95,  # MJ/kg
                water_consumption=0.15,  # L/kg
                waste_generation=0.05,  # kg/kg
                recyclability=0.3,
                renewable_content=0.0,
                transportation_distance=50,  # km
                transportation_mode='truck',
                manufacturing_location='local',
                disposal_method='recycling'
            ),
            'steel': LCAMaterial(
                material_id='steel',
                name='Steel',
                category='structural',
                embodied_carbon=1.85,  # kg CO2/kg
                embodied_energy=25.0,  # MJ/kg
                water_consumption=0.8,  # L/kg
                waste_generation=0.1,  # kg/kg
                recyclability=0.9,
                renewable_content=0.0,
                transportation_distance=200,  # km
                transportation_mode='truck',
                manufacturing_location='regional',
                disposal_method='recycling'
            ),
            'brick': LCAMaterial(
                material_id='brick',
                name='Brick',
                category='structural',
                embodied_carbon=0.24,  # kg CO2/kg
                embodied_energy=2.5,  # MJ/kg
                water_consumption=0.3,  # L/kg
                waste_generation=0.02,  # kg/kg
                recyclability=0.8,
                renewable_content=0.0,
                transportation_distance=30,  # km
                transportation_mode='truck',
                manufacturing_location='local',
                disposal_method='reuse'
            ),
            'wood': LCAMaterial(
                material_id='wood',
                name='Wood',
                category='structural',
                embodied_carbon=-0.9,  # kg CO2/kg (carbon storage)
                embodied_energy=8.5,  # MJ/kg
                water_consumption=0.2,  # L/kg
                waste_generation=0.01,  # kg/kg
                recyclability=0.95,
                renewable_content=1.0,
                transportation_distance=100,  # km
                transportation_mode='truck',
                manufacturing_location='regional',
                disposal_method='reuse'
            ),
            'glass': LCAMaterial(
                material_id='glass',
                name='Glass',
                category='finishing',
                embodied_carbon=0.85,  # kg CO2/kg
                embodied_energy=15.0,  # MJ/kg
                water_consumption=0.5,  # L/kg
                waste_generation=0.03,  # kg/kg
                recyclability=0.9,
                renewable_content=0.0,
                transportation_distance=150,  # km
                transportation_mode='truck',
                manufacturing_location='regional',
                disposal_method='recycling'
            ),
            'insulation_fiberglass': LCAMaterial(
                material_id='insulation_fiberglass',
                name='Fiberglass Insulation',
                category='insulation',
                embodied_carbon=1.2,  # kg CO2/kg
                embodied_energy=28.0,  # MJ/kg
                water_consumption=0.1,  # L/kg
                waste_generation=0.02,  # kg/kg
                recyclability=0.7,
                renewable_content=0.0,
                transportation_distance=300,  # km
                transportation_mode='truck',
                manufacturing_location='national',
                disposal_method='landfill'
            ),
            'insulation_rockwool': LCAMaterial(
                material_id='insulation_rockwool',
                name='Rockwool Insulation',
                category='insulation',
                embodied_carbon=1.4,  # kg CO2/kg
                embodied_energy=16.0,  # MJ/kg
                water_consumption=0.05,  # L/kg
                waste_generation=0.01,  # kg/kg
                recyclability=0.8,
                renewable_content=0.0,
                transportation_distance=250,  # km
                transportation_mode='truck',
                manufacturing_location='national',
                disposal_method='recycling'
            ),
            'aluminum': LCAMaterial(
                material_id='aluminum',
                name='Aluminum',
                category='finishing',
                embodied_carbon=8.24,  # kg CO2/kg
                embodied_energy=155.0,  # MJ/kg
                water_consumption=1.5,  # L/kg
                waste_generation=0.2,  # kg/kg
                recyclability=0.95,
                renewable_content=0.0,
                transportation_distance=500,  # km
                transportation_mode='truck',
                manufacturing_location='national',
                disposal_method='recycling'
            ),
            'vinyl': LCAMaterial(
                material_id='vinyl',
                name='Vinyl',
                category='finishing',
                embodied_carbon=2.5,  # kg CO2/kg
                embodied_energy=45.0,  # MJ/kg
                water_consumption=0.3,  # L/kg
                waste_generation=0.05,  # kg/kg
                recyclability=0.6,
                renewable_content=0.0,
                transportation_distance=400,  # km
                transportation_mode='truck',
                manufacturing_location='national',
                disposal_method='incineration'
            )
        }
        return materials
    
    def _initialize_processes_lca(self) -> Dict[str, LCAProcess]:
        """Initialize LCA data for building processes"""
        processes = {
            'construction': LCAProcess(
                process_id='construction',
                name='Construction',
                category='construction',
                carbon_emissions=50.0,  # kg CO2/m²
                energy_consumption=200.0,  # MJ/m²
                water_consumption=100.0,  # L/m²
                waste_generation=20.0,  # kg/m²
                duration=1.0,  # years
                frequency=1.0,  # times per year
                impact_factors={
                    'GWP': 50.0,  # Global Warming Potential
                    'AP': 0.3,    # Acidification Potential
                    'EP': 0.1,    # Eutrophication Potential
                    'ODP': 0.001, # Ozone Depletion Potential
                    'POCP': 0.2   # Photochemical Ozone Creation Potential
                }
            ),
            'operation_heating': LCAProcess(
                process_id='operation_heating',
                name='Heating Operation',
                category='operation',
                carbon_emissions=25.0,  # kg CO2/m²/year
                energy_consumption=150.0,  # MJ/m²/year
                water_consumption=10.0,  # L/m²/year
                waste_generation=2.0,  # kg/m²/year
                duration=50.0,  # years
                frequency=1.0,  # times per year
                impact_factors={
                    'GWP': 25.0,
                    'AP': 0.15,
                    'EP': 0.05,
                    'ODP': 0.0001,
                    'POCP': 0.1
                }
            ),
            'operation_cooling': LCAProcess(
                process_id='operation_cooling',
                name='Cooling Operation',
                category='operation',
                carbon_emissions=15.0,  # kg CO2/m²/year
                energy_consumption=100.0,  # MJ/m²/year
                water_consumption=5.0,  # L/m²/year
                waste_generation=1.0,  # kg/m²/year
                duration=50.0,  # years
                frequency=1.0,  # times per year
                impact_factors={
                    'GWP': 15.0,
                    'AP': 0.1,
                    'EP': 0.03,
                    'ODP': 0.0001,
                    'POCP': 0.05
                }
            ),
            'operation_lighting': LCAProcess(
                process_id='operation_lighting',
                name='Lighting Operation',
                category='operation',
                carbon_emissions=8.0,  # kg CO2/m²/year
                energy_consumption=50.0,  # MJ/m²/year
                water_consumption=0.0,  # L/m²/year
                waste_generation=0.5,  # kg/m²/year
                duration=50.0,  # years
                frequency=1.0,  # times per year
                impact_factors={
                    'GWP': 8.0,
                    'AP': 0.05,
                    'EP': 0.02,
                    'ODP': 0.0001,
                    'POCP': 0.03
                }
            ),
            'maintenance': LCAProcess(
                process_id='maintenance',
                name='Maintenance',
                category='maintenance',
                carbon_emissions=5.0,  # kg CO2/m²/year
                energy_consumption=20.0,  # MJ/m²/year
                water_consumption=2.0,  # L/m²/year
                waste_generation=3.0,  # kg/m²/year
                duration=50.0,  # years
                frequency=0.2,  # times per year
                impact_factors={
                    'GWP': 5.0,
                    'AP': 0.03,
                    'EP': 0.01,
                    'ODP': 0.0001,
                    'POCP': 0.02
                }
            ),
            'demolition': LCAProcess(
                process_id='demolition',
                name='Demolition',
                category='demolition',
                carbon_emissions=10.0,  # kg CO2/m²
                energy_consumption=50.0,  # MJ/m²
                water_consumption=5.0,  # L/m²
                waste_generation=100.0,  # kg/m²
                duration=1.0,  # years
                frequency=0.02,  # times per year (50-year lifetime)
                impact_factors={
                    'GWP': 10.0,
                    'AP': 0.06,
                    'EP': 0.02,
                    'ODP': 0.0001,
                    'POCP': 0.04
                }
            )
        }
        return processes
    
    def _initialize_impact_categories(self) -> Dict[str, Dict[str, Any]]:
        """Initialize impact categories and their characteristics"""
        return {
            'GWP': {
                'name': 'Global Warming Potential',
                'unit': 'kg CO2-eq',
                'time_horizon': 100,  # years
                'description': 'Contribution to climate change'
            },
            'AP': {
                'name': 'Acidification Potential',
                'unit': 'kg SO2-eq',
                'time_horizon': 0,
                'description': 'Contribution to acidification'
            },
            'EP': {
                'name': 'Eutrophication Potential',
                'unit': 'kg PO4-eq',
                'time_horizon': 0,
                'description': 'Contribution to eutrophication'
            },
            'ODP': {
                'name': 'Ozone Depletion Potential',
                'unit': 'kg CFC-11-eq',
                'time_horizon': 0,
                'description': 'Contribution to ozone depletion'
            },
            'POCP': {
                'name': 'Photochemical Ozone Creation Potential',
                'unit': 'kg C2H4-eq',
                'time_horizon': 0,
                'description': 'Contribution to smog formation'
            }
        }
    
    def calculate_material_lca(self, material_id: str, quantity: float, 
                             area: float = 1.0) -> Dict[str, float]:
        """Calculate LCA impacts for a material"""
        material = self.materials_lca[material_id]
        
        # Calculate impacts per unit area
        embodied_carbon = material.embodied_carbon * quantity / area
        embodied_energy = material.embodied_energy * quantity / area
        water_consumption = material.water_consumption * quantity / area
        waste_generation = material.waste_generation * quantity / area
        
        # Calculate transportation impacts
        transport_carbon = self._calculate_transport_carbon(
            material.transportation_distance, 
            material.transportation_mode, 
            quantity
        ) / area
        
        # Calculate disposal impacts
        disposal_carbon = self._calculate_disposal_carbon(
            material.disposal_method, 
            quantity
        ) / area
        
        return {
            'embodied_carbon': embodied_carbon,
            'embodied_energy': embodied_energy,
            'water_consumption': water_consumption,
            'waste_generation': waste_generation,
            'transport_carbon': transport_carbon,
            'disposal_carbon': disposal_carbon,
            'total_carbon': embodied_carbon + transport_carbon + disposal_carbon,
            'recyclability': material.recyclability,
            'renewable_content': material.renewable_content
        }
    
    def _calculate_transport_carbon(self, distance: float, mode: str, quantity: float) -> float:
        """Calculate transportation carbon emissions"""
        # Carbon intensity by transportation mode (kg CO2/tonne-km)
        carbon_intensity = {
            'truck': 0.1,
            'rail': 0.03,
            'ship': 0.01,
            'air': 0.5
        }
        
        return carbon_intensity.get(mode, 0.1) * distance * quantity / 1000
    
    def _calculate_disposal_carbon(self, method: str, quantity: float) -> float:
        """Calculate disposal carbon emissions"""
        # Carbon intensity by disposal method (kg CO2/kg)
        carbon_intensity = {
            'landfill': 0.1,
            'incineration': 0.5,
            'recycling': -0.2,  # Carbon credit
            'reuse': -0.5  # Higher carbon credit
        }
        
        return carbon_intensity.get(method, 0.1) * quantity
    
    def calculate_process_lca(self, process_id: str, area: float, 
                            duration: float = 1.0) -> Dict[str, float]:
        """Calculate LCA impacts for a process"""
        process = self.processes_lca[process_id]
        
        # Calculate impacts over duration
        carbon_emissions = process.carbon_emissions * area * duration
        energy_consumption = process.energy_consumption * area * duration
        water_consumption = process.water_consumption * area * duration
        waste_generation = process.waste_generation * area * duration
        
        # Calculate impact factors
        impact_factors = {}
        for impact, value in process.impact_factors.items():
            impact_factors[impact] = value * area * duration
        
        return {
            'carbon_emissions': carbon_emissions,
            'energy_consumption': energy_consumption,
            'water_consumption': water_consumption,
            'waste_generation': waste_generation,
            'impact_factors': impact_factors
        }
    
    def generate_building_lca(self, building_dna: Dict[str, Any], 
                            assessment_period: int = 50) -> LCABuilding:
        """Generate complete LCA assessment for a building"""
        print("Generating building LCA assessment...")
        
        building_id = building_dna['metadata']['building_id']
        building_type = building_dna['metadata']['building_type']
        total_area = building_dna['geometry']['total_area']
        
        # Initialize totals
        total_embodied_carbon = 0
        total_embodied_energy = 0
        total_operational_carbon = 0
        total_operational_energy = 0
        total_water_consumption = 0
        total_waste_generation = 0
        
        materials_breakdown = {}
        processes_breakdown = {}
        impact_categories = {cat: 0 for cat in self.impact_categories.keys()}
        
        # Calculate material impacts
        wall_assembly = building_dna['construction_materials']['wall_assembly']
        for layer in wall_assembly['layers']:
            material_id = layer['material_name'].lower().replace(' ', '_')
            if material_id in self.materials_lca:
                quantity = layer['thickness_m'] * 1000  # Convert to kg/m²
                material_impacts = self.calculate_material_lca(material_id, quantity, 1.0)
                
                total_embodied_carbon += material_impacts['total_carbon'] * total_area
                total_embodied_energy += material_impacts['embodied_energy'] * total_area
                total_water_consumption += material_impacts['water_consumption'] * total_area
                total_waste_generation += material_impacts['waste_generation'] * total_area
                
                materials_breakdown[material_id] = {
                    'carbon': material_impacts['total_carbon'] * total_area,
                    'energy': material_impacts['embodied_energy'] * total_area,
                    'water': material_impacts['water_consumption'] * total_area,
                    'waste': material_impacts['waste_generation'] * total_area
                }
        
        # Calculate operational impacts
        operational_processes = ['operation_heating', 'operation_cooling', 'operation_lighting']
        for process_id in operational_processes:
            process_impacts = self.calculate_process_lca(process_id, total_area, assessment_period)
            
            total_operational_carbon += process_impacts['carbon_emissions']
            total_operational_energy += process_impacts['energy_consumption']
            total_water_consumption += process_impacts['water_consumption']
            total_waste_generation += process_impacts['waste_generation']
            
            processes_breakdown[process_id] = {
                'carbon': process_impacts['carbon_emissions'],
                'energy': process_impacts['energy_consumption'],
                'water': process_impacts['water_consumption'],
                'waste': process_impacts['waste_generation']
            }
            
            # Add impact factors
            for impact, value in process_impacts['impact_factors'].items():
                impact_categories[impact] += value
        
        # Calculate maintenance impacts
        maintenance_impacts = self.calculate_process_lca('maintenance', total_area, assessment_period)
        total_operational_carbon += maintenance_impacts['carbon_emissions']
        total_operational_energy += maintenance_impacts['energy_consumption']
        total_water_consumption += maintenance_impacts['water_consumption']
        total_waste_generation += maintenance_impacts['waste_generation']
        
        processes_breakdown['maintenance'] = {
            'carbon': maintenance_impacts['carbon_emissions'],
            'energy': maintenance_impacts['energy_consumption'],
            'water': maintenance_impacts['water_consumption'],
            'waste': maintenance_impacts['waste_generation']
        }
        
        # Calculate demolition impacts
        demolition_impacts = self.calculate_process_lca('demolition', total_area, 1.0)
        total_embodied_carbon += demolition_impacts['carbon_emissions']
        total_embodied_energy += demolition_impacts['energy_consumption']
        total_water_consumption += demolition_impacts['water_consumption']
        total_waste_generation += demolition_impacts['waste_generation']
        
        processes_breakdown['demolition'] = {
            'carbon': demolition_impacts['carbon_emissions'],
            'energy': demolition_impacts['energy_consumption'],
            'water': demolition_impacts['water_consumption'],
            'waste': demolition_impacts['waste_generation']
        }
        
        # Calculate intensities
        carbon_intensity = (total_embodied_carbon + total_operational_carbon) / (total_area * assessment_period)
        energy_intensity = (total_embodied_energy + total_operational_energy) / (total_area * assessment_period)
        water_intensity = total_water_consumption / (total_area * assessment_period)
        waste_intensity = total_waste_generation / (total_area * assessment_period)
        
        # Generate recommendations
        recommendations = self._generate_lca_recommendations(
            building_type, carbon_intensity, energy_intensity, 
            materials_breakdown, processes_breakdown
        )
        
        return LCABuilding(
            building_id=building_id,
            assessment_date=datetime.now().isoformat(),
            assessment_period=assessment_period,
            total_embodied_carbon=total_embodied_carbon,
            total_operational_carbon=total_operational_carbon,
            total_embodied_energy=total_embodied_energy,
            total_operational_energy=total_operational_energy,
            total_water_consumption=total_water_consumption,
            total_waste_generation=total_waste_generation,
            carbon_intensity=carbon_intensity,
            energy_intensity=energy_intensity,
            water_intensity=water_intensity,
            waste_intensity=waste_intensity,
            materials_breakdown=materials_breakdown,
            processes_breakdown=processes_breakdown,
            impact_categories=impact_categories,
            recommendations=recommendations
        )
    
    def _generate_lca_recommendations(self, building_type: str, carbon_intensity: float, 
                                    energy_intensity: float, materials_breakdown: Dict[str, Any],
                                    processes_breakdown: Dict[str, Any]) -> List[str]:
        """Generate LCA-based recommendations for building improvement"""
        recommendations = []
        
        # Carbon intensity recommendations
        if carbon_intensity > 100:  # kg CO2/m²/year
            recommendations.append("Consider using low-carbon materials to reduce embodied carbon")
            recommendations.append("Implement renewable energy systems to reduce operational carbon")
        elif carbon_intensity > 50:
            recommendations.append("Optimize building envelope to reduce heating and cooling loads")
            recommendations.append("Consider material substitutions for high-impact components")
        
        # Energy intensity recommendations
        if energy_intensity > 500:  # MJ/m²/year
            recommendations.append("Improve building insulation to reduce energy consumption")
            recommendations.append("Upgrade HVAC systems for better efficiency")
        elif energy_intensity > 300:
            recommendations.append("Implement energy management systems")
            recommendations.append("Consider renewable energy integration")
        
        # Material-specific recommendations
        high_carbon_materials = [mat for mat, data in materials_breakdown.items() 
                               if data['carbon'] > 1000]
        if high_carbon_materials:
            recommendations.append(f"Consider replacing high-carbon materials: {', '.join(high_carbon_materials)}")
        
        # Process-specific recommendations
        if processes_breakdown.get('operation_heating', {}).get('carbon', 0) > 5000:
            recommendations.append("Improve heating system efficiency or consider heat pumps")
        
        if processes_breakdown.get('operation_cooling', {}).get('carbon', 0) > 3000:
            recommendations.append("Optimize cooling systems and consider passive cooling strategies")
        
        # Building type specific recommendations
        if building_type == 'residential':
            recommendations.append("Consider solar panels for renewable energy generation")
            recommendations.append("Implement water-saving fixtures and appliances")
        elif building_type in ['commercial', 'office']:
            recommendations.append("Implement smart building controls for energy optimization")
            recommendations.append("Consider green roof or wall systems")
        elif building_type == 'industrial':
            recommendations.append("Implement waste heat recovery systems")
            recommendations.append("Consider process optimization for energy efficiency")
        
        return recommendations
    
    def export_lca_data(self, lca_building: LCABuilding, filename: str = None) -> str:
        """Export LCA data to JSON"""
        if filename is None:
            filename = f"lca_assessment_{lca_building.building_id}.json"
        
        data = asdict(lca_building)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"LCA data exported to: {filename}")
        return filename
    
    def visualize_lca_impacts(self, lca_building: LCABuilding, save_path: str = None):
        """Visualize LCA impacts"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Carbon Emissions', 'Energy Consumption', 
                          'Water Consumption', 'Waste Generation'],
            specs=[[{'type': 'pie'}, {'type': 'pie'}],
                   [{'type': 'pie'}, {'type': 'pie'}]]
        )
        
        # Carbon emissions breakdown
        carbon_data = {
            'Embodied': lca_building.total_embodied_carbon,
            'Operational': lca_building.total_operational_carbon
        }
        
        fig.add_trace(go.Pie(
            labels=list(carbon_data.keys()),
            values=list(carbon_data.values()),
            name="Carbon Emissions"
        ), row=1, col=1)
        
        # Energy consumption breakdown
        energy_data = {
            'Embodied': lca_building.total_embodied_energy,
            'Operational': lca_building.total_operational_energy
        }
        
        fig.add_trace(go.Pie(
            labels=list(energy_data.keys()),
            values=list(energy_data.values()),
            name="Energy Consumption"
        ), row=1, col=2)
        
        # Water consumption by process
        water_data = {}
        for process, data in lca_building.processes_breakdown.items():
            water_data[process] = data['water']
        
        fig.add_trace(go.Pie(
            labels=list(water_data.keys()),
            values=list(water_data.values()),
            name="Water Consumption"
        ), row=2, col=1)
        
        # Waste generation by process
        waste_data = {}
        for process, data in lca_building.processes_breakdown.items():
            waste_data[process] = data['waste']
        
        fig.add_trace(go.Pie(
            labels=list(waste_data.keys()),
            values=list(waste_data.values()),
            name="Waste Generation"
        ), row=2, col=2)
        
        fig.update_layout(
            title=f"LCA Impact Assessment - Building {lca_building.building_id}",
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"LCA visualization saved to: {save_path}")
        else:
            fig.show()
    
    def generate_lca_comparison(self, lca_buildings: List[LCABuilding], save_path: str = None):
        """Generate LCA comparison between buildings"""
        # Prepare data
        building_ids = [b.building_id for b in lca_buildings]
        carbon_intensities = [b.carbon_intensity for b in lca_buildings]
        energy_intensities = [b.energy_intensity for b in lca_buildings]
        water_intensities = [b.water_intensity for b in lca_buildings]
        waste_intensities = [b.waste_intensity for b in lca_buildings]
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Carbon Intensity',
            x=building_ids,
            y=carbon_intensities,
            yaxis='y',
            offsetgroup=1
        ))
        
        fig.add_trace(go.Bar(
            name='Energy Intensity',
            x=building_ids,
            y=energy_intensities,
            yaxis='y2',
            offsetgroup=2
        ))
        
        fig.add_trace(go.Bar(
            name='Water Intensity',
            x=building_ids,
            y=water_intensities,
            yaxis='y3',
            offsetgroup=3
        ))
        
        fig.add_trace(go.Bar(
            name='Waste Intensity',
            x=building_ids,
            y=waste_intensities,
            yaxis='y4',
            offsetgroup=4
        ))
        
        # Update layout
        fig.update_layout(
            title='LCA Impact Comparison Between Buildings',
            xaxis_title='Building ID',
            yaxis=dict(title='Carbon Intensity (kg CO2/m²/year)', side='left'),
            yaxis2=dict(title='Energy Intensity (MJ/m²/year)', side='right'),
            yaxis3=dict(title='Water Intensity (L/m²/year)', side='left'),
            yaxis4=dict(title='Waste Intensity (kg/m²/year)', side='right'),
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"LCA comparison saved to: {save_path}")
        else:
            fig.show()

def main():
    """Main function to demonstrate LCA integration"""
    print("Life-Cycle Assessment Integration")
    print("=" * 40)
    
    # Initialize LCA integration
    lca = LCAIntegration(seed=42)
    
    # Create sample building DNA
    from building_dna_generator import BuildingDNAGenerator
    dna_generator = BuildingDNAGenerator(seed=42)
    building_dna = dna_generator.generate_complete_building_dna('office')
    
    # Generate LCA assessment
    lca_building = lca.generate_building_lca(building_dna, assessment_period=50)
    
    # Export LCA data
    lca.export_lca_data(lca_building)
    
    # Generate visualizations
    lca.visualize_lca_impacts(lca_building, 'lca_impacts_visualization.html')
    
    print(f"\nLCA Assessment Complete:")
    print(f"Carbon Intensity: {lca_building.carbon_intensity:.2f} kg CO2/m²/year")
    print(f"Energy Intensity: {lca_building.energy_intensity:.2f} MJ/m²/year")
    print(f"Water Intensity: {lca_building.water_intensity:.2f} L/m²/year")
    print(f"Waste Intensity: {lca_building.waste_intensity:.2f} kg/m²/year")
    print(f"\nRecommendations: {len(lca_building.recommendations)} generated")

if __name__ == "__main__":
    main()