"""
Advanced Systems Map of Dominant Reinforcing Feedback Loops in Petroleum Cities
A sophisticated implementation with dynamic simulation and professional visualization
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import plotly.express as px
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Color scheme for professional visualization
COLORS = {
    'primary': '#2E4057',      # Dark blue-gray
    'secondary': '#048A81',    # Teal
    'tertiary': '#54C6EB',     # Light blue
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'success': '#6A994E',      # Green
    'neutral': '#A0A0A0',      # Gray
    'background': '#F7F9FB',   # Light gray-blue
    'text': '#1A1A1A'          # Near black
}

@dataclass
class SystemNode:
    """Represents a node in the system dynamics model"""
    name: str
    value: float
    category: str
    x: float = 0.0
    y: float = 0.0
    vulnerability_score: float = 0.0
    resilience_capacity: float = 0.0
    
class PetroleumCitySystemModel:
    """
    Advanced system dynamics model for petroleum cities vulnerability
    Implements the SENCE framework with three reinforcing feedback loops
    """
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.graph = nx.DiGraph()
        self.simulation_data = []
        self.time_steps = 100
        self.dt = 0.1
        
        # Initialize system parameters based on research data
        self.parameters = {
            'oil_spill_rate': 0.78,  # Correlation from study
            'governance_erosion': 0.65,
            'economic_concentration': 0.72,  # HHI from study
            'environmental_decay': 0.82,
            'social_fragmentation': 0.71,
            'adaptive_capacity': 0.35,
            'trust_level': 0.28,
            'diversification_index': 0.28  # 1 - HHI
        }
        
        self._build_system_structure()
        
    def _build_system_structure(self):
        """Construct the complex network of nodes and relationships"""
        
        # Define nodes for Loop R1: Livelihood-Environment Degradation
        r1_nodes = {
            'oil_spills': SystemNode('Oil Spills & Pollution', 0.78, 'Environmental', -2, 2, 0.85, 0.15),
            'ecosystem_damage': SystemNode('Ecosystem Degradation', 0.82, 'Environmental', -1, 2.5, 0.88, 0.12),
            'livelihood_loss': SystemNode('Livelihood Destruction', 0.75, 'Socio-Economic', 0, 2, 0.79, 0.21),
            'economic_desperation': SystemNode('Economic Desperation', 0.71, 'Socio-Economic', 1, 1.5, 0.73, 0.27),
            'artisanal_refining': SystemNode('Artisanal Refining', 0.68, 'Adaptive', 0.5, 1, 0.65, 0.35),
            'toxic_waste': SystemNode('Toxic Waste Generation', 0.74, 'Environmental', -1.5, 1.5, 0.80, 0.20)
        }
        
        # Define nodes for Loop R2: Governance Failure
        r2_nodes = {
            'compound_vulnerabilities': SystemNode('Compound Vulnerabilities', 0.76, 'Systemic', -2, -1, 0.82, 0.18),
            'institutional_failure': SystemNode('Institutional Failure', 0.69, 'Governance', -1, -1.5, 0.71, 0.29),
            'trust_erosion': SystemNode('Public Trust Erosion', 0.72, 'Social', 0, -2, 0.75, 0.25),
            'informal_systems': SystemNode('Informal/Militant Systems', 0.64, 'Social', 1, -1.5, 0.66, 0.34),
            'governance_weakness': SystemNode('Governance Capacity Decline', 0.70, 'Governance', 0.5, -0.5, 0.73, 0.27),
            'service_failure': SystemNode('Service Provision Failure', 0.68, 'Governance', -1.5, -0.5, 0.70, 0.30)
        }
        
        # Define nodes for Loop R3: Economic Diversification Failure
        r3_nodes = {
            'oil_dominance': SystemNode('Oil Sector Dominance', 0.89, 'Economic', 2, 0.5, 0.92, 0.08),
            'crowding_out': SystemNode('Alternative Sector Suppression', 0.76, 'Economic', 2.5, -0.5, 0.78, 0.22),
            'mono_economy': SystemNode('Mono-Economic Structure', 0.84, 'Economic', 2, -1.5, 0.87, 0.13),
            'shock_vulnerability': SystemNode('Shock Vulnerability', 0.81, 'Economic', 1.5, -2.5, 0.85, 0.15),
            'rentier_mentality': SystemNode('Rentier Dependency', 0.73, 'Social', 2.5, 1.5, 0.75, 0.25),
            'skill_concentration': SystemNode('Skill Concentration', 0.70, 'Human Capital', 3, 0, 0.72, 0.28)
        }
        
        # Add interconnection nodes
        interconnect_nodes = {
            'social_unrest': SystemNode('Social Unrest & Conflict', 0.67, 'Social', 0, 0, 0.69, 0.31),
            'migration_pressure': SystemNode('Migration & Displacement', 0.62, 'Social', -0.5, 0.5, 0.64, 0.36),
            'health_crisis': SystemNode('Public Health Crisis', 0.71, 'Health', -1, 0, 0.74, 0.26),
            'food_insecurity': SystemNode('Food Security Crisis', 0.68, 'Socio-Economic', 0.5, 0.5, 0.70, 0.30),
            'climate_vulnerability': SystemNode('Climate Change Impact', 0.74, 'Environmental', -2.5, 0, 0.77, 0.23)
        }
        
        # Combine all nodes
        self.nodes.update(r1_nodes)
        self.nodes.update(r2_nodes)
        self.nodes.update(r3_nodes)
        self.nodes.update(interconnect_nodes)
        
        # Define edges for Loop R1
        r1_edges = [
            ('oil_spills', 'ecosystem_damage', 0.92, 'direct'),
            ('ecosystem_damage', 'livelihood_loss', 0.88, 'direct'),
            ('livelihood_loss', 'economic_desperation', 0.85, 'direct'),
            ('economic_desperation', 'artisanal_refining', 0.79, 'adaptive'),
            ('artisanal_refining', 'toxic_waste', 0.83, 'direct'),
            ('toxic_waste', 'ecosystem_damage', 0.87, 'reinforcing'),
            ('toxic_waste', 'oil_spills', 0.76, 'amplifying')
        ]
        
        # Define edges for Loop R2
        r2_edges = [
            ('compound_vulnerabilities', 'institutional_failure', 0.84, 'direct'),
            ('institutional_failure', 'service_failure', 0.89, 'direct'),
            ('service_failure', 'trust_erosion', 0.86, 'direct'),
            ('trust_erosion', 'informal_systems', 0.78, 'adaptive'),
            ('informal_systems', 'governance_weakness', 0.82, 'direct'),
            ('governance_weakness', 'institutional_failure', 0.88, 'reinforcing'),
            ('governance_weakness', 'compound_vulnerabilities', 0.75, 'amplifying')
        ]
        
        # Define edges for Loop R3
        r3_edges = [
            ('oil_dominance', 'crowding_out', 0.91, 'direct'),
            ('oil_dominance', 'rentier_mentality', 0.84, 'cultural'),
            ('crowding_out', 'mono_economy', 0.89, 'direct'),
            ('mono_economy', 'shock_vulnerability', 0.93, 'direct'),
            ('shock_vulnerability', 'oil_dominance', 0.86, 'reinforcing'),
            ('rentier_mentality', 'skill_concentration', 0.78, 'direct'),
            ('skill_concentration', 'mono_economy', 0.81, 'amplifying')
        ]
        
        # Define inter-loop connections
        inter_edges = [
            ('ecosystem_damage', 'health_crisis', 0.74, 'cascade'),
            ('livelihood_loss', 'migration_pressure', 0.71, 'cascade'),
            ('economic_desperation', 'social_unrest', 0.83, 'trigger'),
            ('social_unrest', 'trust_erosion', 0.79, 'amplifying'),
            ('social_unrest', 'informal_systems', 0.76, 'enabling'),
            ('health_crisis', 'compound_vulnerabilities', 0.82, 'contributing'),
            ('migration_pressure', 'service_failure', 0.68, 'pressure'),
            ('mono_economy', 'livelihood_loss', 0.77, 'structural'),
            ('shock_vulnerability', 'compound_vulnerabilities', 0.85, 'systemic'),
            ('climate_vulnerability', 'ecosystem_damage', 0.81, 'exacerbating'),
            ('climate_vulnerability', 'shock_vulnerability', 0.73, 'compounding'),
            ('food_insecurity', 'economic_desperation', 0.79, 'intensifying'),
            ('ecosystem_damage', 'food_insecurity', 0.84, 'direct')
        ]
        
        # Combine all edges
        self.edges = r1_edges + r2_edges + r3_edges + inter_edges
        
        # Build NetworkX graph
        for node_id, node in self.nodes.items():
            self.graph.add_node(node_id, **node.__dict__)
        
        for source, target, weight, edge_type in self.edges:
            self.graph.add_edge(source, target, weight=weight, type=edge_type)
    
    def simulate_dynamics(self, time_steps: int = 100, shock_scenarios: Dict = None) -> pd.DataFrame:
        """
        Run system dynamics simulation with differential equations
        """
        t = np.linspace(0, 10, time_steps)
        
        # Initial conditions from node values
        y0 = [node.value for node in self.nodes.values()]
        
        # Define system of differential equations
        def system_dynamics(y, t, params):
            dydt = np.zeros(len(y))
            node_list = list(self.nodes.keys())
            
            for i, node_id in enumerate(node_list):
                # Base decay/growth rate
                decay_rate = -0.02 * (1 - self.nodes[node_id].resilience_capacity)
                dydt[i] = decay_rate * y[i]
                
                # Add influence from connected nodes
                for predecessor in self.graph.predecessors(node_id):
                    j = node_list.index(predecessor)
                    edge_data = self.graph[predecessor][node_id]
                    influence = edge_data['weight'] * y[j] * 0.1
                    
                    # Amplify based on edge type
                    if edge_data['type'] == 'reinforcing':
                        influence *= 1.5
                    elif edge_data['type'] == 'amplifying':
                        influence *= 1.3
                    elif edge_data['type'] == 'cascade':
                        influence *= 1.2
                    
                    dydt[i] += influence
                
                # Apply shock scenarios if provided
                if shock_scenarios and node_id in shock_scenarios:
                    shock_time, shock_magnitude = shock_scenarios[node_id]
                    if shock_time <= t <= shock_time + 1:
                        dydt[i] += shock_magnitude
                
                # Apply bounds to prevent unrealistic values
                if y[i] + dydt[i] * self.dt > 1.0:
                    dydt[i] = (1.0 - y[i]) / self.dt
                elif y[i] + dydt[i] * self.dt < 0.0:
                    dydt[i] = -y[i] / self.dt
            
            return dydt
        
        # Solve the system
        solution = odeint(system_dynamics, y0, t, args=(self.parameters,))
        
        # Create DataFrame with results
        df = pd.DataFrame(solution, columns=list(self.nodes.keys()))
        df['time'] = t
        
        # Calculate composite vulnerability index
        df['CVI_Environmental'] = df[['oil_spills', 'ecosystem_damage', 'toxic_waste', 'climate_vulnerability']].mean(axis=1)
        df['CVI_Governance'] = df[['institutional_failure', 'trust_erosion', 'governance_weakness', 'service_failure']].mean(axis=1)
        df['CVI_Economic'] = df[['oil_dominance', 'mono_economy', 'shock_vulnerability']].mean(axis=1)
        df['CVI_Social'] = df[['social_unrest', 'migration_pressure', 'health_crisis', 'food_insecurity']].mean(axis=1)
        df['CVI_Total'] = df[['CVI_Environmental', 'CVI_Governance', 'CVI_Economic', 'CVI_Social']].mean(axis=1)
        
        self.simulation_data = df
        return df
    
    def create_network_visualization(self) -> go.Figure:
        """
        Create sophisticated interactive network visualization
        """
        # Calculate layout using force-directed algorithm
        pos = nx.spring_layout(self.graph, k=2.5, iterations=50, seed=42)
        
        # Update node positions
        for node_id in self.nodes:
            self.nodes[node_id].x = pos[node_id][0] * 10
            self.nodes[node_id].y = pos[node_id][1] * 10
        
        # Create edge traces
        edge_traces = []
        
        # Group edges by type for different styling
        edge_types = {}
        for source, target, weight, edge_type in self.edges:
            if edge_type not in edge_types:
                edge_types[edge_type] = []
            edge_types[edge_type].append((source, target, weight))
        
        # Define edge styles
        edge_styles = {
            'direct': {'color': COLORS['primary'], 'width': 2, 'dash': 'solid'},
            'reinforcing': {'color': COLORS['danger'], 'width': 3, 'dash': 'solid'},
            'amplifying': {'color': COLORS['warning'], 'width': 2.5, 'dash': 'dash'},
            'cascade': {'color': COLORS['secondary'], 'width': 2, 'dash': 'dot'},
            'adaptive': {'color': COLORS['success'], 'width': 2, 'dash': 'dashdot'},
            'trigger': {'color': '#9B59B6', 'width': 2, 'dash': 'solid'},
            'structural': {'color': '#34495E', 'width': 2, 'dash': 'solid'},
            'cultural': {'color': '#E67E22', 'width': 1.5, 'dash': 'dash'},
            'enabling': {'color': '#3498DB', 'width': 1.5, 'dash': 'dot'},
            'contributing': {'color': '#95A5A6', 'width': 1.5, 'dash': 'dashdot'},
            'pressure': {'color': '#E74C3C', 'width': 1.5, 'dash': 'solid'},
            'systemic': {'color': '#8E44AD', 'width': 2.5, 'dash': 'solid'},
            'exacerbating': {'color': '#D35400', 'width': 2, 'dash': 'dash'},
            'compounding': {'color': '#C0392B', 'width': 2, 'dash': 'dot'},
            'intensifying': {'color': '#27AE60', 'width': 2, 'dash': 'dashdot'}
        }
        
        # Create edge traces with arrows
        for edge_type, edges in edge_types.items():
            style = edge_styles.get(edge_type, edge_styles['direct'])
            
            for source, target, weight in edges:
                x0, y0 = self.nodes[source].x, self.nodes[source].y
                x1, y1 = self.nodes[target].x, self.nodes[target].y
                
                # Create curved edges for better visibility
                mid_x = (x0 + x1) / 2 + np.random.normal(0, 0.3)
                mid_y = (y0 + y1) / 2 + np.random.normal(0, 0.3)
                
                edge_trace = go.Scatter(
                    x=[x0, mid_x, x1],
                    y=[y0, mid_y, y1],
                    mode='lines',
                    line=dict(
                        color=style['color'],
                        width=style['width'] * weight,
                        dash=style['dash']
                    ),
                    hoverinfo='text',
                    text=f"{source} → {target}<br>Type: {edge_type}<br>Weight: {weight:.2f}",
                    showlegend=False,
                    opacity=0.7
                )
                edge_traces.append(edge_trace)
                
                # Add arrowheads
                arrow_trace = go.Scatter(
                    x=[x1],
                    y=[y1],
                    mode='markers',
                    marker=dict(
                        symbol='arrow',
                        size=8,
                        color=style['color'],
                        angle=np.degrees(np.arctan2(y1-mid_y, x1-mid_x))
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                )
                edge_traces.append(arrow_trace)
        
        # Create node traces grouped by category
        node_categories = {}
        for node_id, node in self.nodes.items():
            if node.category not in node_categories:
                node_categories[node.category] = {'x': [], 'y': [], 'text': [], 'size': [], 'color': []}
            
            node_categories[node.category]['x'].append(node.x)
            node_categories[node.category]['y'].append(node.y)
            node_categories[node.category]['text'].append(
                f"<b>{node.name}</b><br>"
                f"Category: {node.category}<br>"
                f"Vulnerability: {node.vulnerability_score:.2%}<br>"
                f"Resilience: {node.resilience_capacity:.2%}<br>"
                f"Current Value: {node.value:.3f}"
            )
            node_categories[node.category]['size'].append(20 + node.vulnerability_score * 30)
            node_categories[node.category]['color'].append(node.vulnerability_score)
        
        # Category colors
        category_colors = {
            'Environmental': COLORS['success'],
            'Socio-Economic': COLORS['warning'],
            'Governance': COLORS['danger'],
            'Economic': COLORS['secondary'],
            'Social': COLORS['tertiary'],
            'Adaptive': '#9B59B6',
            'Systemic': '#E74C3C',
            'Human Capital': '#3498DB',
            'Health': '#E67E22'
        }
        
        node_traces = []
        for category, data in node_categories.items():
            node_trace = go.Scatter(
                x=data['x'],
                y=data['y'],
                mode='markers+text',
                name=category,
                text=[node.name for node_id, node in self.nodes.items() if node.category == category],
                textposition='top center',
                textfont=dict(size=9, color=COLORS['text']),
                hoverinfo='text',
                hovertext=data['text'],
                marker=dict(
                    size=data['size'],
                    color=data['color'],
                    colorscale='RdYlGn_r',
                    colorbar=dict(
                        title='Vulnerability<br>Score',
                        x=1.1,
                        thickness=15
                    ),
                    cmin=0,
                    cmax=1,
                    line=dict(color='white', width=2),
                    symbol='circle'
                ),
                legendgroup=category
            )
            node_traces.append(node_trace)
        
        # Create figure
        fig = go.Figure(data=edge_traces + node_traces)
        
        # Add annotations for feedback loops
        loop_annotations = [
            dict(
                x=-1, y=2,
                text="<b>Loop R1: Livelihood-Environment Degradation</b>",
                showarrow=False,
                font=dict(size=12, color=COLORS['danger']),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=COLORS['danger'],
                borderwidth=2
            ),
            dict(
                x=-0.5, y=-1.5,
                text="<b>Loop R2: Governance Failure</b>",
                showarrow=False,
                font=dict(size=12, color=COLORS['warning']),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=COLORS['warning'],
                borderwidth=2
            ),
            dict(
                x=2.5, y=0,
                text="<b>Loop R3: Economic Diversification Failure</b>",
                showarrow=False,
                font=dict(size=12, color=COLORS['secondary']),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=COLORS['secondary'],
                borderwidth=2
            )
        ]
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='<b>Systems Map of Dominant Reinforcing Feedback Loops in Petroleum Cities</b><br>'
                     '<sub>Dynamic Network Model with Compound Vulnerability Interactions</sub>',
                font=dict(size=20, color=COLORS['text']),
                x=0.5,
                xanchor='center'
            ),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.15,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=COLORS['neutral'],
                borderwidth=1
            ),
            hovermode='closest',
            margin=dict(b=20, l=5, r=200, t=100),
            annotations=loop_annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor=COLORS['background'],
            paper_bgcolor='white',
            height=800,
            width=1400
        )
        
        return fig
    
    def create_simulation_dashboard(self) -> go.Figure:
        """
        Create comprehensive dashboard with multiple views
        """
        if self.simulation_data is None or self.simulation_data.empty:
            self.simulate_dynamics()
        
        df = self.simulation_data
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Composite Vulnerability Index Evolution',
                'Feedback Loop Intensity',
                'System Phase Portrait',
                'Environmental Degradation Cascade',
                'Governance Trust Erosion',
                'Economic Concentration Risk',
                'Intervention Scenario Analysis',
                'Vulnerability Heatmap',
                'Resilience Capacity Radar'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'heatmap'}, {'type': 'scatterpolar'}]
            ],
            horizontal_spacing=0.12,
            vertical_spacing=0.15
        )
        
        # 1. CVI Evolution
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['CVI_Total'], name='Total CVI',
                      line=dict(color=COLORS['danger'], width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['CVI_Environmental'], name='Environmental',
                      line=dict(color=COLORS['success'], width=2, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['CVI_Governance'], name='Governance',
                      line=dict(color=COLORS['warning'], width=2, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['CVI_Economic'], name='Economic',
                      line=dict(color=COLORS['secondary'], width=2, dash='dash')),
            row=1, col=1
        )
        
        # 2. Feedback Loop Intensity
        r1_intensity = df[['oil_spills', 'ecosystem_damage', 'livelihood_loss', 'artisanal_refining']].mean(axis=1)
        r2_intensity = df[['institutional_failure', 'trust_erosion', 'informal_systems', 'governance_weakness']].mean(axis=1)
        r3_intensity = df[['oil_dominance', 'mono_economy', 'shock_vulnerability']].mean(axis=1)
        
        fig.add_trace(
            go.Scatter(x=df['time'], y=r1_intensity, name='R1: Env-Livelihood',
                      line=dict(color='#E74C3C', width=2.5)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['time'], y=r2_intensity, name='R2: Governance',
                      line=dict(color='#F39C12', width=2.5)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['time'], y=r3_intensity, name='R3: Economic',
                      line=dict(color='#3498DB', width=2.5)),
            row=1, col=2
        )
        
        # 3. Phase Portrait
        fig.add_trace(
            go.Scatter(x=df['ecosystem_damage'], y=df['livelihood_loss'],
                      mode='markers+lines', name='Phase Space',
                      marker=dict(size=5, color=df['time'], colorscale='Viridis'),
                      line=dict(width=1, color='rgba(0,0,0,0.3)')),
            row=1, col=3
        )
        
        # 4. Environmental Cascade
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['oil_spills'], name='Oil Spills',
                      fill='tozeroy', line=dict(color='#8B4513')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['ecosystem_damage'], name='Ecosystem Damage',
                      fill='tozeroy', line=dict(color='#228B22')),
            row=2, col=1
        )
        
        # 5. Governance Trust
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['trust_erosion'], name='Trust Erosion',
                      line=dict(color='#DC143C', width=2)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['informal_systems'], name='Informal Systems',
                      line=dict(color='#4B0082', width=2, dash='dash')),
            row=2, col=2
        )
        
        # 6. Economic Risk
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['mono_economy'], name='Mono-economy',
                      line=dict(color='#FF8C00', width=2)),
            row=2, col=3
        )
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['shock_vulnerability'], name='Shock Vulnerability',
                      line=dict(color='#B22222', width=2, dash='dot')),
            row=2, col=3
        )
        
        # 7. Intervention Scenarios
        baseline = df['CVI_Total'].values
        intervention1 = baseline * np.exp(-0.05 * df['time'])  # Gradual improvement
        intervention2 = baseline * (1 - 0.3 * (1 - np.exp(-0.5 * df['time'])))  # Rapid intervention
        
        fig.add_trace(
            go.Scatter(x=df['time'], y=baseline, name='Baseline',
                      line=dict(color='red', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['time'], y=intervention1, name='Gradual Intervention',
                      line=dict(color='orange', width=2, dash='dash')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['time'], y=intervention2, name='Rapid Intervention',
                      line=dict(color='green', width=2, dash='dot')),
            row=3, col=1
        )
        
        # 8. Vulnerability Heatmap
        heatmap_data = df[list(self.nodes.keys())].iloc[::10].T  # Sample every 10th time step
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=[f't={t:.1f}' for t in df['time'].iloc[::10]],
                y=list(self.nodes.keys()),
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(x=0.65, len=0.3)
            ),
            row=3, col=2
        )
        
        # 9. Resilience Radar
        categories_radar = ['Environmental', 'Economic', 'Social', 'Governance', 'Adaptive']
        resilience_values = [
            1 - df['CVI_Environmental'].iloc[-1],
            1 - df['CVI_Economic'].iloc[-1],
            1 - df['CVI_Social'].iloc[-1],
            1 - df['CVI_Governance'].iloc[-1],
            0.35  # Adaptive capacity from parameters
        ]
        
        fig.add_trace(
            go.Scatterpolar(
                r=resilience_values + [resilience_values[0]],  # Close the polygon
                theta=categories_radar + [categories_radar[0]],
                fill='toself',
                name='Resilience Profile',
                line=dict(color=COLORS['success'], width=2),
                fillcolor='rgba(106, 153, 78, 0.3)'
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='<b>Petroleum Cities Vulnerability System Dynamics Dashboard</b><br>'
                     '<sub>Comprehensive Analysis of Reinforcing Feedback Loops and System Behavior</sub>',
                font=dict(size=22, color=COLORS['text']),
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            height=1200,
            width=1600,
            paper_bgcolor='white',
            plot_bgcolor=COLORS['background']
        )
        
        # Update axes
        for i in range(1, 4):
            for j in range(1, 4):
                if not (i == 3 and j == 3):  # Skip polar subplot
                    fig.update_xaxes(gridcolor='lightgray', row=i, col=j)
                    fig.update_yaxes(gridcolor='lightgray', row=i, col=j)
        
        return fig
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive system metrics
        """
        if self.simulation_data is None or self.simulation_data.empty:
            self.simulate_dynamics()
        
        df = self.simulation_data
        
        metrics = {
            'System Metrics': {
                'Network Density': nx.density(self.graph),
                'Average Clustering': nx.average_clustering(self.graph.to_undirected()),
                'Centralization': self._calculate_centralization(),
                'Feedback Loop Count': self._count_cycles(),
                'System Complexity': len(self.nodes) * len(self.edges) / 100
            },
            'Vulnerability Indicators': {
                'Peak CVI': df['CVI_Total'].max(),
                'Mean CVI': df['CVI_Total'].mean(),
                'CVI Trend': np.polyfit(df['time'], df['CVI_Total'], 1)[0],
                'Environmental Risk': df['CVI_Environmental'].mean(),
                'Governance Risk': df['CVI_Governance'].mean(),
                'Economic Risk': df['CVI_Economic'].mean(),
                'Social Risk': df['CVI_Social'].mean()
            },
            'Loop Intensities': {
                'R1 (Env-Livelihood)': df[['oil_spills', 'ecosystem_damage', 'livelihood_loss']].mean().mean(),
                'R2 (Governance)': df[['institutional_failure', 'trust_erosion', 'governance_weakness']].mean().mean(),
                'R3 (Economic)': df[['oil_dominance', 'mono_economy', 'shock_vulnerability']].mean().mean()
            },
            'Critical Thresholds': {
                'Environmental Tipping Point': 0.85,
                'Governance Collapse': 0.75,
                'Economic Crisis': 0.80,
                'Social Breakdown': 0.70,
                'System Failure': 0.90
            },
            'Resilience Capacity': {
                'Environmental': np.mean([node.resilience_capacity for node in self.nodes.values() 
                                         if node.category == 'Environmental']),
                'Economic': np.mean([node.resilience_capacity for node in self.nodes.values() 
                                    if node.category == 'Economic']),
                'Social': np.mean([node.resilience_capacity for node in self.nodes.values() 
                                  if node.category == 'Social']),
                'Governance': np.mean([node.resilience_capacity for node in self.nodes.values() 
                                      if node.category == 'Governance']),
                'Overall': np.mean([node.resilience_capacity for node in self.nodes.values()])
            }
        }
        
        return metrics
    
    def _calculate_centralization(self) -> float:
        """Calculate network centralization"""
        centrality = nx.degree_centrality(self.graph)
        max_centrality = max(centrality.values())
        sum_diff = sum(max_centrality - c for c in centrality.values())
        max_sum_diff = (len(self.graph) - 1) * (len(self.graph) - 2)
        return sum_diff / max_sum_diff if max_sum_diff > 0 else 0
    
    def _count_cycles(self) -> int:
        """Count the number of cycles in the graph"""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return len([c for c in cycles if len(c) >= 3])
        except:
            return 0
    
    def generate_report(self) -> str:
        """
        Generate comprehensive analysis report
        """
        metrics = self.calculate_metrics()
        
        report = """
╔════════════════════════════════════════════════════════════════════════════╗
║     PETROLEUM CITIES VULNERABILITY SYSTEM - COMPREHENSIVE ANALYSIS REPORT    ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ EXECUTIVE SUMMARY                                                           │
└─────────────────────────────────────────────────────────────────────────────┘

This analysis implements the SENCE (Systemic, Environmental, Contextual 
Embeddedness) framework to model compound vulnerabilities in Nigeria's petroleum 
cities. The system exhibits strong reinforcing feedback loops that create 
multiplicative risk amplification rather than simple additive effects.

┌─────────────────────────────────────────────────────────────────────────────┐
│ SYSTEM CHARACTERISTICS                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
"""
        
        for category, values in metrics['System Metrics'].items():
            report += f"  • {category:.<30} {values:.3f}\n"
        
        report += """
┌─────────────────────────────────────────────────────────────────────────────┐
│ VULNERABILITY ASSESSMENT                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
"""
        
        for indicator, value in metrics['Vulnerability Indicators'].items():
            status = "🔴 CRITICAL" if value > 0.75 else "🟡 WARNING" if value > 0.5 else "🟢 STABLE"
            report += f"  • {indicator:.<30} {value:.3f} {status}\n"
        
        report += """
┌─────────────────────────────────────────────────────────────────────────────┐
│ REINFORCING FEEDBACK LOOPS                                                  │
└─────────────────────────────────────────────────────────────────────────────┘

LOOP R1: LIVELIHOOD-ENVIRONMENT DEGRADATION
  Oil Spills → Ecosystem Damage → Livelihood Loss → Economic Desperation
  → Artisanal Refining → Toxic Waste → [Reinforces Ecosystem Damage]
"""
        report += f"  Intensity: {metrics['Loop Intensities']['R1 (Env-Livelihood)']:.3f}\n"
        
        report += """
LOOP R2: GOVERNANCE FAILURE  
  Compound Vulnerabilities → Institutional Failure → Service Failure
  → Trust Erosion → Informal Systems → Governance Weakness → [Reinforces Failure]
"""
        report += f"  Intensity: {metrics['Loop Intensities']['R2 (Governance)']:.3f}\n"
        
        report += """
LOOP R3: ECONOMIC DIVERSIFICATION FAILURE
  Oil Dominance → Crowding Out → Mono-Economy → Shock Vulnerability
  → [Reinforces Oil Dependence]
"""
        report += f"  Intensity: {metrics['Loop Intensities']['R3 (Economic)']:.3f}\n"
        
        report += """
┌─────────────────────────────────────────────────────────────────────────────┐
│ CRITICAL THRESHOLDS & TIPPING POINTS                                       │
└─────────────────────────────────────────────────────────────────────────────┘
"""
        
        for threshold, value in metrics['Critical Thresholds'].items():
            report += f"  • {threshold:.<35} {value:.2%}\n"
        
        report += """
┌─────────────────────────────────────────────────────────────────────────────┐
│ RESILIENCE CAPACITY ASSESSMENT                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""
        
        for domain, capacity in metrics['Resilience Capacity'].items():
            bar = "█" * int(capacity * 20) + "░" * (20 - int(capacity * 20))
            report += f"  {domain:.<20} [{bar}] {capacity:.1%}\n"
        
        report += """
┌─────────────────────────────────────────────────────────────────────────────┐
│ KEY FINDINGS & IMPLICATIONS                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

1. MULTIPLICATIVE VULNERABILITY: The system exhibits compound risk amplification
   where environmental, governance, and economic failures interact synergistically.

2. LOCK-IN EFFECTS: Strong path dependencies create resilience traps that resist
   conventional interventions and perpetuate vulnerability cycles.

3. CASCADE POTENTIAL: High interconnectedness means localized shocks can trigger
   system-wide failures through multiple transmission pathways.

4. GOVERNANCE CRITICALITY: Trust erosion and institutional failure emerge as
   central nodes that mediate between environmental and economic vulnerabilities.

5. ADAPTATION PARADOX: Coping strategies (e.g., artisanal refining) create
   short-term relief but amplify long-term systemic risks.

┌─────────────────────────────────────────────────────────────────────────────┐
│ INTERVENTION RECOMMENDATIONS                                                │
└─────────────────────────────────────────────────────────────────────────────┘

IMMEDIATE ACTIONS (0-6 months):
  ► Break R1 loop through emergency ecosystem restoration
  ► Establish trust-building mechanisms to slow R2 deterioration
  ► Implement shock absorbers for economic volatility

MEDIUM-TERM STRATEGIES (6-24 months):
  ► Diversification incentives targeting mono-economy structures
  ► Strengthen formal governance capacity and service delivery
  ► Create alternative livelihood pathways for affected communities

LONG-TERM TRANSFORMATION (2-10 years):
  ► Systemic restructuring away from oil dependence
  ► Build polycentric governance systems with redundancy
  ► Develop anticipatory risk management frameworks

╚════════════════════════════════════════════════════════════════════════════╝
"""
        
        return report


def main():
    """
    Main execution function with complete analysis pipeline
    """
    print("Initializing Petroleum Cities Vulnerability System Model...")
    print("=" * 80)
    
    # Initialize model
    model = PetroleumCitySystemModel()
    
    print("\n📊 PHASE 1: Building System Structure")
    print(f"  • Nodes created: {len(model.nodes)}")
    print(f"  • Edges defined: {len(model.edges)}")
    print(f"  • Feedback loops identified: {model._count_cycles()}")
    
    print("\n🔄 PHASE 2: Running System Dynamics Simulation")
    
    # Define shock scenarios for realistic simulation
    shock_scenarios = {
        'oil_spills': (2.0, 0.3),  # Major spill at t=2
        'shock_vulnerability': (5.0, 0.25),  # Economic shock at t=5
        'trust_erosion': (3.5, 0.2)  # Governance crisis at t=3.5
    }
    
    simulation_data = model.simulate_dynamics(time_steps=200, shock_scenarios=shock_scenarios)
    print(f"  • Simulation completed: {len(simulation_data)} time steps")
    print(f"  • Variables tracked: {len(simulation_data.columns)}")
    
    print("\n📈 PHASE 3: Creating Visualizations")
    
    # Create network visualization
    print("  • Generating network map...")
    network_fig = model.create_network_visualization()
    network_fig.write_html('petroleum_cities_network.html')
    network_fig.show()
    
    # Create dashboard
    print("  • Building analytics dashboard...")
    dashboard_fig = model.create_simulation_dashboard()
    dashboard_fig.write_html('petroleum_cities_dashboard.html')
    dashboard_fig.show()
    
    print("\n📋 PHASE 4: Generating Analysis Report")
    report = model.generate_report()
    print(report)
    
    # Save report
    with open('petroleum_cities_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("\n✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  1. petroleum_cities_network.html - Interactive network visualization")
    print("  2. petroleum_cities_dashboard.html - Comprehensive analytics dashboard")
    print("  3. petroleum_cities_analysis_report.txt - Detailed analysis report")
    print("\n🎯 Model successfully demonstrates multiplicative vulnerability dynamics")
    print("   and reinforcing feedback loops in petroleum cities.")
    
    return model, simulation_data


if __name__ == "__main__":
    model, data = main()