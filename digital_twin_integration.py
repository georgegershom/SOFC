#!/usr/bin/env python3
"""
Digital Twin Framework Integration for Retrofit Decision Engine
Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization

This module provides integration capabilities for:
- Real-time IoT data integration
- Deep reinforcement learning optimization
- Multi-objective optimization algorithms
- Digital twin state management
- Predictive analytics and decision support
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path
from abc import ABC, abstractmethod
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IoTSensorData:
    """Structure for IoT sensor data"""
    sensor_id: str
    timestamp: datetime
    sensor_type: str
    value: float
    unit: str
    location: str
    quality_score: float = 1.0

@dataclass
class BuildingState:
    """Current state of the building digital twin"""
    timestamp: datetime
    indoor_temperature_c: float
    outdoor_temperature_c: float
    humidity_percent: float
    occupancy_count: int
    energy_consumption_kw: float
    hvac_status: Dict
    lighting_status: Dict
    renewable_generation_kw: float = 0.0
    battery_soc_percent: float = 0.0

@dataclass
class OptimizationObjective:
    """Multi-objective optimization objective definition"""
    name: str
    weight: float
    minimize: bool
    target_value: Optional[float] = None
    tolerance: float = 0.05

@dataclass
class RetrofitRecommendation:
    """Retrofit recommendation from the decision engine"""
    recommendation_id: str
    timestamp: datetime
    measures: List[str]
    priority_score: float
    estimated_savings: Dict
    implementation_timeline: str
    confidence_level: float
    reasoning: List[str]

class IoTDataManager:
    """Manages IoT sensor data collection and processing"""
    
    def __init__(self):
        self.sensors: Dict[str, IoTSensorData] = {}
        self.data_buffer: List[IoTSensorData] = []
        self.max_buffer_size = 10000
        
    def register_sensor(self, sensor_id: str, sensor_type: str, location: str) -> None:
        """Register a new IoT sensor"""
        logger.info(f"Registering sensor {sensor_id} of type {sensor_type} at {location}")
        
    async def collect_sensor_data(self, sensor_id: str) -> IoTSensorData:
        """Simulate real-time sensor data collection"""
        # Simulate sensor data based on type
        sensor_types = {
            'temperature': {'range': (18, 26), 'unit': '°C'},
            'humidity': {'range': (30, 70), 'unit': '%'},
            'occupancy': {'range': (0, 50), 'unit': 'people'},
            'power': {'range': (10, 100), 'unit': 'kW'},
            'co2': {'range': (400, 1200), 'unit': 'ppm'},
            'light_level': {'range': (100, 800), 'unit': 'lux'}
        }
        
        # Determine sensor type from ID
        sensor_type = 'temperature'  # Default
        for stype in sensor_types.keys():
            if stype in sensor_id.lower():
                sensor_type = stype
                break
        
        # Generate realistic data with some noise
        base_value = np.random.uniform(*sensor_types[sensor_type]['range'])
        noise = np.random.normal(0, base_value * 0.05)  # 5% noise
        value = max(0, base_value + noise)
        
        # Add time-based variations
        hour = datetime.now().hour
        if sensor_type == 'occupancy':
            # Higher occupancy during business hours
            if 8 <= hour <= 18:
                value *= 1.5
            else:
                value *= 0.2
        elif sensor_type == 'power':
            # Higher power consumption during business hours
            if 8 <= hour <= 18:
                value *= 1.3
            else:
                value *= 0.6
        
        sensor_data = IoTSensorData(
            sensor_id=sensor_id,
            timestamp=datetime.now(),
            sensor_type=sensor_type,
            value=round(value, 2),
            unit=sensor_types[sensor_type]['unit'],
            location=f"zone_{sensor_id.split('_')[-1] if '_' in sensor_id else '1'}",
            quality_score=np.random.uniform(0.85, 1.0)
        )
        
        # Add to buffer
        self.data_buffer.append(sensor_data)
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer.pop(0)
        
        return sensor_data
    
    def get_recent_data(self, sensor_type: str = None, hours: int = 24) -> List[IoTSensorData]:
        """Get recent sensor data"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_data = [
            data for data in self.data_buffer
            if data.timestamp >= cutoff_time
        ]
        
        if sensor_type:
            filtered_data = [
                data for data in filtered_data
                if data.sensor_type == sensor_type
            ]
        
        return filtered_data
    
    def calculate_data_quality_metrics(self) -> Dict:
        """Calculate data quality metrics for IoT sensors"""
        if not self.data_buffer:
            return {"error": "No data available"}
        
        recent_data = self.get_recent_data(hours=1)
        
        metrics = {
            'total_sensors': len(set(data.sensor_id for data in recent_data)),
            'data_points_last_hour': len(recent_data),
            'average_quality_score': np.mean([data.quality_score for data in recent_data]),
            'sensor_coverage': {},
            'data_freshness_minutes': {}
        }
        
        # Calculate per-sensor metrics
        for sensor_type in set(data.sensor_type for data in recent_data):
            type_data = [data for data in recent_data if data.sensor_type == sensor_type]
            metrics['sensor_coverage'][sensor_type] = len(type_data)
            
            if type_data:
                latest_timestamp = max(data.timestamp for data in type_data)
                minutes_old = (datetime.now() - latest_timestamp).total_seconds() / 60
                metrics['data_freshness_minutes'][sensor_type] = round(minutes_old, 1)
        
        return metrics

class DigitalTwinEngine:
    """Core digital twin engine for building state management"""
    
    def __init__(self, building_id: str):
        self.building_id = building_id
        self.current_state: Optional[BuildingState] = None
        self.state_history: List[BuildingState] = []
        self.iot_manager = IoTDataManager()
        self.prediction_models: Dict[str, Callable] = {}
        
    async def update_building_state(self) -> BuildingState:
        """Update building state from IoT sensors"""
        # Collect data from various sensors
        sensor_tasks = [
            self.iot_manager.collect_sensor_data('temp_indoor_01'),
            self.iot_manager.collect_sensor_data('temp_outdoor_01'),
            self.iot_manager.collect_sensor_data('humidity_01'),
            self.iot_manager.collect_sensor_data('occupancy_01'),
            self.iot_manager.collect_sensor_data('power_total_01'),
        ]
        
        sensor_data = await asyncio.gather(*sensor_tasks)
        
        # Extract values from sensor data
        indoor_temp = next((d.value for d in sensor_data if d.sensor_type == 'temperature' and 'indoor' in d.sensor_id), 22.0)
        outdoor_temp = next((d.value for d in sensor_data if d.sensor_type == 'temperature' and 'outdoor' in d.sensor_id), 20.0)
        humidity = next((d.value for d in sensor_data if d.sensor_type == 'humidity'), 50.0)
        occupancy = int(next((d.value for d in sensor_data if d.sensor_type == 'occupancy'), 10))
        power = next((d.value for d in sensor_data if d.sensor_type == 'power'), 50.0)
        
        # Create building state
        self.current_state = BuildingState(
            timestamp=datetime.now(),
            indoor_temperature_c=indoor_temp,
            outdoor_temperature_c=outdoor_temp,
            humidity_percent=humidity,
            occupancy_count=occupancy,
            energy_consumption_kw=power,
            hvac_status={'mode': 'auto', 'setpoint': 22.0, 'fan_speed': 'medium'},
            lighting_status={'zones_on': 8, 'total_zones': 12, 'dimming_level': 0.8},
            renewable_generation_kw=np.random.uniform(0, 15),  # Simulated solar
            battery_soc_percent=np.random.uniform(20, 90)  # Simulated battery
        )
        
        # Add to history
        self.state_history.append(self.current_state)
        if len(self.state_history) > 1000:  # Keep last 1000 states
            self.state_history.pop(0)
        
        return self.current_state
    
    def predict_future_state(self, hours_ahead: int = 24) -> List[BuildingState]:
        """Predict future building states using ML models"""
        if not self.state_history:
            logger.warning("No historical data available for prediction")
            return []
        
        predictions = []
        current_time = datetime.now()
        
        # Simple prediction model based on historical patterns
        for hour in range(1, hours_ahead + 1):
            future_time = current_time + timedelta(hours=hour)
            
            # Predict based on time of day patterns
            hour_of_day = future_time.hour
            
            # Temperature prediction (simplified)
            if 6 <= hour_of_day <= 18:  # Daytime
                temp_factor = 1.0 + 0.1 * np.sin((hour_of_day - 6) * np.pi / 12)
            else:  # Nighttime
                temp_factor = 0.9
            
            predicted_indoor_temp = self.current_state.indoor_temperature_c * temp_factor
            
            # Occupancy prediction
            if 8 <= hour_of_day <= 18:  # Business hours
                predicted_occupancy = int(self.current_state.occupancy_count * 1.2)
            else:
                predicted_occupancy = max(1, int(self.current_state.occupancy_count * 0.1))
            
            # Energy consumption prediction
            occupancy_factor = predicted_occupancy / max(self.current_state.occupancy_count, 1)
            predicted_power = self.current_state.energy_consumption_kw * occupancy_factor
            
            predicted_state = BuildingState(
                timestamp=future_time,
                indoor_temperature_c=round(predicted_indoor_temp, 1),
                outdoor_temperature_c=self.current_state.outdoor_temperature_c,  # Simplified
                humidity_percent=self.current_state.humidity_percent,
                occupancy_count=predicted_occupancy,
                energy_consumption_kw=round(predicted_power, 1),
                hvac_status=self.current_state.hvac_status.copy(),
                lighting_status=self.current_state.lighting_status.copy(),
                renewable_generation_kw=self._predict_solar_generation(future_time),
                battery_soc_percent=self.current_state.battery_soc_percent
            )
            
            predictions.append(predicted_state)
        
        return predictions
    
    def _predict_solar_generation(self, timestamp: datetime) -> float:
        """Predict solar generation based on time of day"""
        hour = timestamp.hour
        
        if 6 <= hour <= 18:  # Daylight hours
            # Simple sine wave for solar generation
            solar_factor = np.sin((hour - 6) * np.pi / 12)
            max_generation = 15.0  # kW
            return round(max_generation * solar_factor * np.random.uniform(0.8, 1.0), 1)
        else:
            return 0.0
    
    def detect_anomalies(self) -> List[Dict]:
        """Detect anomalies in building performance"""
        if len(self.state_history) < 10:
            return []
        
        anomalies = []
        recent_states = self.state_history[-10:]
        
        # Temperature anomaly detection
        temps = [state.indoor_temperature_c for state in recent_states]
        temp_mean = np.mean(temps)
        temp_std = np.std(temps)
        
        if self.current_state.indoor_temperature_c > temp_mean + 2 * temp_std:
            anomalies.append({
                'type': 'temperature_high',
                'severity': 'medium',
                'value': self.current_state.indoor_temperature_c,
                'expected_range': (temp_mean - temp_std, temp_mean + temp_std),
                'timestamp': self.current_state.timestamp
            })
        
        # Energy consumption anomaly detection
        powers = [state.energy_consumption_kw for state in recent_states]
        power_mean = np.mean(powers)
        power_std = np.std(powers)
        
        if self.current_state.energy_consumption_kw > power_mean + 2 * power_std:
            anomalies.append({
                'type': 'energy_consumption_high',
                'severity': 'high',
                'value': self.current_state.energy_consumption_kw,
                'expected_range': (power_mean - power_std, power_mean + power_std),
                'timestamp': self.current_state.timestamp
            })
        
        return anomalies

class MultiObjectiveOptimizer:
    """Multi-objective optimization for retrofit decisions"""
    
    def __init__(self):
        self.objectives: List[OptimizationObjective] = []
        self.constraints: List[Dict] = []
        
    def add_objective(self, name: str, weight: float, minimize: bool = True, 
                     target_value: Optional[float] = None) -> None:
        """Add an optimization objective"""
        objective = OptimizationObjective(
            name=name,
            weight=weight,
            minimize=minimize,
            target_value=target_value
        )
        self.objectives.append(objective)
        logger.info(f"Added objective: {name} (weight: {weight}, minimize: {minimize})")
    
    def add_constraint(self, name: str, constraint_type: str, value: float) -> None:
        """Add an optimization constraint"""
        constraint = {
            'name': name,
            'type': constraint_type,  # 'max', 'min', 'equal'
            'value': value
        }
        self.constraints.append(constraint)
        logger.info(f"Added constraint: {name} {constraint_type} {value}")
    
    def evaluate_solution(self, solution: Dict) -> Dict:
        """Evaluate a retrofit solution against objectives"""
        scores = {}
        total_score = 0
        
        for objective in self.objectives:
            if objective.name in solution:
                value = solution[objective.name]
                
                # Normalize score (simplified)
                if objective.target_value:
                    score = 1.0 - abs(value - objective.target_value) / objective.target_value
                else:
                    # Use relative scoring
                    score = 1.0 / (1.0 + abs(value)) if objective.minimize else value
                
                # Apply weight
                weighted_score = score * objective.weight
                scores[objective.name] = {
                    'raw_value': value,
                    'score': score,
                    'weighted_score': weighted_score
                }
                total_score += weighted_score
        
        return {
            'total_score': total_score,
            'objective_scores': scores,
            'feasible': self._check_constraints(solution)
        }
    
    def _check_constraints(self, solution: Dict) -> bool:
        """Check if solution satisfies all constraints"""
        for constraint in self.constraints:
            if constraint['name'] in solution:
                value = solution[constraint['name']]
                
                if constraint['type'] == 'max' and value > constraint['value']:
                    return False
                elif constraint['type'] == 'min' and value < constraint['value']:
                    return False
                elif constraint['type'] == 'equal' and abs(value - constraint['value']) > 0.01:
                    return False
        
        return True
    
    def optimize_portfolio(self, candidate_solutions: List[Dict]) -> Dict:
        """Find optimal retrofit portfolio from candidates"""
        best_solution = None
        best_score = float('-inf')
        
        evaluations = []
        
        for solution in candidate_solutions:
            evaluation = self.evaluate_solution(solution)
            evaluations.append({
                'solution': solution,
                'evaluation': evaluation
            })
            
            if evaluation['feasible'] and evaluation['total_score'] > best_score:
                best_score = evaluation['total_score']
                best_solution = solution
        
        return {
            'optimal_solution': best_solution,
            'optimal_score': best_score,
            'all_evaluations': evaluations,
            'num_feasible_solutions': sum(1 for e in evaluations if e['evaluation']['feasible'])
        }

class ReinforcementLearningAgent:
    """Deep reinforcement learning agent for retrofit optimization"""
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        self.episode_count = 0
        
    def get_state_key(self, state: BuildingState) -> str:
        """Convert building state to discrete key for Q-table"""
        # Discretize continuous values
        temp_bin = int(state.indoor_temperature_c / 2) * 2  # 2-degree bins
        occupancy_bin = int(state.occupancy_count / 10) * 10  # 10-person bins
        power_bin = int(state.energy_consumption_kw / 10) * 10  # 10kW bins
        
        return f"temp_{temp_bin}_occ_{occupancy_bin}_power_{power_bin}"
    
    def get_available_actions(self) -> List[str]:
        """Get available retrofit actions"""
        return [
            'no_action',
            'adjust_hvac_setpoint',
            'optimize_lighting',
            'activate_demand_response',
            'schedule_maintenance',
            'recommend_insulation_upgrade',
            'recommend_window_upgrade',
            'recommend_hvac_upgrade'
        ]
    
    def select_action(self, state: BuildingState) -> str:
        """Select action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        available_actions = self.get_available_actions()
        
        # Initialize Q-values for new state
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in available_actions}
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(available_actions)
        else:
            # Exploit: best known action
            q_values = self.q_table[state_key]
            return max(q_values.keys(), key=lambda a: q_values[a])
    
    def update_q_value(self, state: BuildingState, action: str, reward: float, 
                      next_state: BuildingState) -> None:
        """Update Q-value using Q-learning algorithm"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-tables if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.get_available_actions()}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in self.get_available_actions()}
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values())
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
    
    def calculate_reward(self, state: BuildingState, action: str, 
                        next_state: BuildingState) -> float:
        """Calculate reward for state transition"""
        reward = 0.0
        
        # Energy efficiency reward
        energy_reduction = state.energy_consumption_kw - next_state.energy_consumption_kw
        reward += energy_reduction * 10  # $10 per kW saved
        
        # Comfort reward
        temp_comfort = 1.0 - abs(next_state.indoor_temperature_c - 22.0) / 5.0  # Target 22°C
        reward += temp_comfort * 5
        
        # Occupancy satisfaction
        if next_state.occupancy_count > 0:
            occupancy_factor = min(next_state.occupancy_count / 50, 1.0)  # Normalize to 50 people
            reward += occupancy_factor * 3
        
        # Action-specific rewards
        if action == 'recommend_insulation_upgrade':
            reward += 20  # High reward for recommending energy-saving measures
        elif action == 'activate_demand_response':
            reward += 15  # Reward for grid-friendly actions
        elif action == 'no_action' and abs(next_state.indoor_temperature_c - 22.0) < 1.0:
            reward += 5  # Reward for maintaining comfort without action
        
        return reward
    
    def train_episode(self, initial_state: BuildingState, 
                     digital_twin: DigitalTwinEngine) -> Dict:
        """Train the agent for one episode"""
        self.episode_count += 1
        total_reward = 0
        actions_taken = []
        
        current_state = initial_state
        
        # Simulate episode for 24 hours (24 steps)
        for step in range(24):
            # Select and execute action
            action = self.select_action(current_state)
            actions_taken.append(action)
            
            # Simulate next state (simplified)
            next_state = self._simulate_action_effect(current_state, action)
            
            # Calculate reward
            reward = self.calculate_reward(current_state, action, next_state)
            total_reward += reward
            
            # Update Q-value
            self.update_q_value(current_state, action, reward, next_state)
            
            # Move to next state
            current_state = next_state
        
        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return {
            'episode': self.episode_count,
            'total_reward': total_reward,
            'actions_taken': actions_taken,
            'final_epsilon': self.epsilon
        }
    
    def _simulate_action_effect(self, state: BuildingState, action: str) -> BuildingState:
        """Simulate the effect of an action on building state"""
        # Create a copy of the current state
        next_state = BuildingState(
            timestamp=state.timestamp + timedelta(hours=1),
            indoor_temperature_c=state.indoor_temperature_c,
            outdoor_temperature_c=state.outdoor_temperature_c,
            humidity_percent=state.humidity_percent,
            occupancy_count=state.occupancy_count,
            energy_consumption_kw=state.energy_consumption_kw,
            hvac_status=state.hvac_status.copy(),
            lighting_status=state.lighting_status.copy(),
            renewable_generation_kw=state.renewable_generation_kw,
            battery_soc_percent=state.battery_soc_percent
        )
        
        # Apply action effects
        if action == 'adjust_hvac_setpoint':
            next_state.hvac_status['setpoint'] = 21.0  # Lower setpoint
            next_state.energy_consumption_kw *= 0.95  # 5% energy reduction
        elif action == 'optimize_lighting':
            next_state.lighting_status['dimming_level'] = 0.7  # Dim lights
            next_state.energy_consumption_kw *= 0.98  # 2% energy reduction
        elif action == 'activate_demand_response':
            next_state.energy_consumption_kw *= 0.85  # 15% load reduction
        elif action == 'recommend_insulation_upgrade':
            next_state.indoor_temperature_c += 0.5  # Better temperature stability
            next_state.energy_consumption_kw *= 0.90  # 10% energy reduction
        
        # Add some natural variation
        next_state.indoor_temperature_c += np.random.normal(0, 0.2)
        next_state.energy_consumption_kw += np.random.normal(0, 2.0)
        next_state.energy_consumption_kw = max(0, next_state.energy_consumption_kw)
        
        return next_state

class RetrofitDecisionEngine:
    """Main decision engine integrating all components"""
    
    def __init__(self, building_id: str):
        self.building_id = building_id
        self.digital_twin = DigitalTwinEngine(building_id)
        self.optimizer = MultiObjectiveOptimizer()
        self.rl_agent = ReinforcementLearningAgent(state_dim=10, action_dim=8)
        self.recommendations_history: List[RetrofitRecommendation] = []
        
        # Initialize optimization objectives
        self._setup_optimization_objectives()
    
    def _setup_optimization_objectives(self) -> None:
        """Setup default optimization objectives"""
        self.optimizer.add_objective('energy_savings_kwh', weight=0.3, minimize=False)
        self.optimizer.add_objective('carbon_reduction_kg_co2e', weight=0.3, minimize=False)
        self.optimizer.add_objective('initial_cost_usd', weight=0.2, minimize=True)
        self.optimizer.add_objective('payback_period_years', weight=0.2, minimize=True)
        
        # Add constraints
        self.optimizer.add_constraint('initial_cost_usd', 'max', 100000)
        self.optimizer.add_constraint('payback_period_years', 'max', 15)
    
    async def generate_recommendations(self) -> RetrofitRecommendation:
        """Generate retrofit recommendations based on current building state"""
        # Update building state
        current_state = await self.digital_twin.update_building_state()
        
        # Get RL agent recommendation
        rl_action = self.rl_agent.select_action(current_state)
        
        # Detect anomalies
        anomalies = self.digital_twin.detect_anomalies()
        
        # Generate candidate retrofit measures
        candidate_measures = self._generate_candidate_measures(current_state, anomalies)
        
        # Evaluate candidates using multi-objective optimization
        optimization_results = self.optimizer.optimize_portfolio(candidate_measures)
        
        # Create recommendation
        recommendation = RetrofitRecommendation(
            recommendation_id=f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            measures=self._extract_measures_from_solution(optimization_results['optimal_solution']),
            priority_score=optimization_results['optimal_score'],
            estimated_savings=self._calculate_estimated_savings(optimization_results['optimal_solution']),
            implementation_timeline=self._determine_implementation_timeline(optimization_results['optimal_solution']),
            confidence_level=self._calculate_confidence_level(current_state, anomalies),
            reasoning=self._generate_reasoning(current_state, anomalies, rl_action, optimization_results)
        )
        
        # Store recommendation
        self.recommendations_history.append(recommendation)
        
        return recommendation
    
    def _generate_candidate_measures(self, state: BuildingState, anomalies: List[Dict]) -> List[Dict]:
        """Generate candidate retrofit measures based on building state and anomalies"""
        candidates = []
        
        # Base efficiency measures
        candidates.append({
            'name': 'LED_lighting_upgrade',
            'energy_savings_kwh': 2000,
            'carbon_reduction_kg_co2e': 800,
            'initial_cost_usd': 15000,
            'payback_period_years': 7.5
        })
        
        candidates.append({
            'name': 'HVAC_optimization',
            'energy_savings_kwh': 5000,
            'carbon_reduction_kg_co2e': 2000,
            'initial_cost_usd': 25000,
            'payback_period_years': 5.0
        })
        
        # Anomaly-driven measures
        for anomaly in anomalies:
            if anomaly['type'] == 'temperature_high':
                candidates.append({
                    'name': 'improved_insulation',
                    'energy_savings_kwh': 8000,
                    'carbon_reduction_kg_co2e': 3200,
                    'initial_cost_usd': 40000,
                    'payback_period_years': 5.0
                })
            elif anomaly['type'] == 'energy_consumption_high':
                candidates.append({
                    'name': 'energy_management_system',
                    'energy_savings_kwh': 12000,
                    'carbon_reduction_kg_co2e': 4800,
                    'initial_cost_usd': 35000,
                    'payback_period_years': 2.9
                })
        
        # State-driven measures
        if state.energy_consumption_kw > 75:  # High energy use
            candidates.append({
                'name': 'solar_pv_system',
                'energy_savings_kwh': 15000,
                'carbon_reduction_kg_co2e': 6000,
                'initial_cost_usd': 60000,
                'payback_period_years': 4.0
            })
        
        if state.occupancy_count > 30:  # High occupancy
            candidates.append({
                'name': 'advanced_ventilation',
                'energy_savings_kwh': 3000,
                'carbon_reduction_kg_co2e': 1200,
                'initial_cost_usd': 20000,
                'payback_period_years': 6.7
            })
        
        return candidates
    
    def _extract_measures_from_solution(self, solution: Optional[Dict]) -> List[str]:
        """Extract measure names from optimization solution"""
        if not solution:
            return ['no_measures_recommended']
        
        return [solution.get('name', 'unknown_measure')]
    
    def _calculate_estimated_savings(self, solution: Optional[Dict]) -> Dict:
        """Calculate estimated savings from solution"""
        if not solution:
            return {'annual_energy_kwh': 0, 'annual_carbon_kg_co2e': 0, 'annual_cost_usd': 0}
        
        return {
            'annual_energy_kwh': solution.get('energy_savings_kwh', 0),
            'annual_carbon_kg_co2e': solution.get('carbon_reduction_kg_co2e', 0),
            'annual_cost_usd': solution.get('energy_savings_kwh', 0) * 0.12  # $0.12/kWh
        }
    
    def _determine_implementation_timeline(self, solution: Optional[Dict]) -> str:
        """Determine implementation timeline based on solution complexity"""
        if not solution:
            return 'immediate'
        
        cost = solution.get('initial_cost_usd', 0)
        
        if cost < 20000:
            return 'short_term_1_3_months'
        elif cost < 50000:
            return 'medium_term_3_6_months'
        else:
            return 'long_term_6_12_months'
    
    def _calculate_confidence_level(self, state: BuildingState, anomalies: List[Dict]) -> float:
        """Calculate confidence level for recommendations"""
        base_confidence = 0.8
        
        # Reduce confidence if many anomalies
        if len(anomalies) > 2:
            base_confidence -= 0.1
        
        # Increase confidence if state is stable
        if len(self.digital_twin.state_history) > 10:
            recent_temps = [s.indoor_temperature_c for s in self.digital_twin.state_history[-10:]]
            if np.std(recent_temps) < 1.0:  # Stable temperature
                base_confidence += 0.1
        
        return min(1.0, max(0.5, base_confidence))
    
    def _generate_reasoning(self, state: BuildingState, anomalies: List[Dict], 
                          rl_action: str, optimization_results: Dict) -> List[str]:
        """Generate reasoning for recommendations"""
        reasoning = []
        
        # State-based reasoning
        if state.energy_consumption_kw > 75:
            reasoning.append("High energy consumption detected - efficiency measures recommended")
        
        if state.indoor_temperature_c > 24:
            reasoning.append("Indoor temperature above comfort range - HVAC optimization suggested")
        
        # Anomaly-based reasoning
        for anomaly in anomalies:
            if anomaly['type'] == 'energy_consumption_high':
                reasoning.append(f"Energy consumption anomaly detected ({anomaly['value']:.1f} kW)")
            elif anomaly['type'] == 'temperature_high':
                reasoning.append(f"Temperature anomaly detected ({anomaly['value']:.1f}°C)")
        
        # RL agent reasoning
        if rl_action != 'no_action':
            reasoning.append(f"AI agent recommends: {rl_action.replace('_', ' ')}")
        
        # Optimization reasoning
        if optimization_results['optimal_solution']:
            reasoning.append(f"Multi-objective optimization selected best solution with score {optimization_results['optimal_score']:.2f}")
        
        return reasoning
    
    async def run_continuous_optimization(self, duration_hours: int = 24) -> Dict:
        """Run continuous optimization for specified duration"""
        logger.info(f"Starting continuous optimization for {duration_hours} hours")
        
        results = {
            'start_time': datetime.now(),
            'duration_hours': duration_hours,
            'recommendations_generated': [],
            'rl_training_episodes': [],
            'performance_metrics': {}
        }
        
        # Training loop
        for hour in range(duration_hours):
            # Update building state
            current_state = await self.digital_twin.update_building_state()
            
            # Generate recommendations every 4 hours
            if hour % 4 == 0:
                recommendation = await self.generate_recommendations()
                results['recommendations_generated'].append(asdict(recommendation))
            
            # Train RL agent
            training_result = self.rl_agent.train_episode(current_state, self.digital_twin)
            results['rl_training_episodes'].append(training_result)
            
            # Simulate time passage
            await asyncio.sleep(0.1)  # Small delay for simulation
        
        # Calculate performance metrics
        if results['rl_training_episodes']:
            rewards = [ep['total_reward'] for ep in results['rl_training_episodes']]
            results['performance_metrics'] = {
                'average_reward': np.mean(rewards),
                'reward_trend': np.polyfit(range(len(rewards)), rewards, 1)[0],  # Linear trend
                'total_recommendations': len(results['recommendations_generated']),
                'average_confidence': np.mean([rec['confidence_level'] for rec in results['recommendations_generated']]) if results['recommendations_generated'] else 0
            }
        
        results['end_time'] = datetime.now()
        logger.info("Continuous optimization completed")
        
        return results

async def main():
    """Example usage of the Digital Twin Framework"""
    
    # Initialize decision engine
    engine = RetrofitDecisionEngine("building_001")
    
    print("=== DIGITAL TWIN RETROFIT DECISION ENGINE ===")
    print("Initializing building digital twin...")
    
    # Update building state
    current_state = await engine.digital_twin.update_building_state()
    print(f"\nCurrent Building State:")
    print(f"  Indoor Temperature: {current_state.indoor_temperature_c:.1f}°C")
    print(f"  Occupancy: {current_state.occupancy_count} people")
    print(f"  Energy Consumption: {current_state.energy_consumption_kw:.1f} kW")
    print(f"  Solar Generation: {current_state.renewable_generation_kw:.1f} kW")
    
    # Generate recommendations
    print("\nGenerating retrofit recommendations...")
    recommendation = await engine.generate_recommendations()
    
    print(f"\n=== RETROFIT RECOMMENDATION ===")
    print(f"ID: {recommendation.recommendation_id}")
    print(f"Priority Score: {recommendation.priority_score:.2f}")
    print(f"Confidence Level: {recommendation.confidence_level:.1%}")
    print(f"Measures: {', '.join(recommendation.measures)}")
    print(f"Implementation Timeline: {recommendation.implementation_timeline}")
    
    print(f"\nEstimated Annual Savings:")
    for key, value in recommendation.estimated_savings.items():
        print(f"  {key}: {value:,.0f}")
    
    print(f"\nReasoning:")
    for reason in recommendation.reasoning:
        print(f"  • {reason}")
    
    # Demonstrate IoT data quality
    print(f"\n=== IOT DATA QUALITY ===")
    data_quality = engine.digital_twin.iot_manager.calculate_data_quality_metrics()
    if 'error' not in data_quality:
        print(f"Active Sensors: {data_quality['total_sensors']}")
        print(f"Data Points (Last Hour): {data_quality['data_points_last_hour']}")
        print(f"Average Quality Score: {data_quality['average_quality_score']:.2f}")
    
    # Demonstrate anomaly detection
    anomalies = engine.digital_twin.detect_anomalies()
    if anomalies:
        print(f"\n=== ANOMALIES DETECTED ===")
        for anomaly in anomalies:
            print(f"  {anomaly['type']}: {anomaly['value']} (severity: {anomaly['severity']})")
    else:
        print(f"\n=== NO ANOMALIES DETECTED ===")
    
    # Demonstrate predictive capabilities
    print(f"\n=== PREDICTIVE ANALYTICS ===")
    predictions = engine.digital_twin.predict_future_state(hours_ahead=6)
    if predictions:
        print("Next 6 hours prediction:")
        for i, pred in enumerate(predictions[:3]):  # Show first 3 hours
            print(f"  Hour +{i+1}: {pred.indoor_temperature_c:.1f}°C, {pred.energy_consumption_kw:.1f}kW, {pred.occupancy_count} people")
    
    # Run short continuous optimization demo
    print(f"\n=== CONTINUOUS OPTIMIZATION DEMO ===")
    print("Running 4-hour optimization simulation...")
    optimization_results = await engine.run_continuous_optimization(duration_hours=4)
    
    metrics = optimization_results['performance_metrics']
    print(f"Average RL Reward: {metrics['average_reward']:.2f}")
    print(f"Reward Trend: {metrics['reward_trend']:.3f}")
    print(f"Recommendations Generated: {metrics['total_recommendations']}")
    print(f"Average Confidence: {metrics['average_confidence']:.1%}")
    
    print(f"\n=== OPTIMIZATION COMPLETE ===")
    print("Digital Twin Framework demonstration completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())