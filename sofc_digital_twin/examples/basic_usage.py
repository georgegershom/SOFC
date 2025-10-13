"""
Basic Usage Examples for SOFC Digital Twin Dataset
Demonstrates how to load, process, and analyze the generated datasets
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from utils.data_processor import SOFCDataProcessor
from utils.visualizer import SOFCVisualizer

def example_1_load_and_explore_dataset1():
    """
    Example 1: Load and explore Dataset 1 (Physics Simulation Data)
    """
    print("="*60)
    print("Example 1: Loading and Exploring Dataset 1")
    print("="*60)
    
    # Initialize data processor
    processor = SOFCDataProcessor()
    
    # Load dataset summary
    summary = processor.load_dataset1_summary()
    print(f"Dataset 1 Summary:")
    print(f"- Total simulations: {summary.get('statistics', {}).get('total_simulations', 'N/A')}")
    print(f"- Success rate: {summary.get('statistics', {}).get('success_rate', 'N/A')}")
    
    # Load first few simulations
    try:
        simulations = processor.load_dataset1_batch(sim_ids=[0, 1, 2])
        print(f"- Loaded {len(simulations)} simulations successfully")
        
        # Examine first simulation
        if simulations:
            sim = simulations[0]
            print(f"\nFirst simulation details:")
            print(f"- Simulation ID: {sim.get('metadata', {}).get('simulation_id', 'N/A')}")
            
            # Operating conditions
            op_cond = sim.get('operating_conditions', {})
            print(f"- Current density: {op_cond.get('current_density', 'N/A'):.3f} A/cm²")
            print(f"- Fuel utilization: {op_cond.get('fuel_utilization', 'N/A'):.3f}")
            print(f"- Inlet fuel temp: {op_cond.get('inlet_fuel_temperature', 'N/A'):.1f} °C")
            
            # Results
            derived = sim.get('derived_quantities', {})
            print(f"- Cell voltage: {derived.get('cell_voltage', 'N/A'):.3f} V")
            print(f"- Max temperature: {derived.get('max_temperature', 'N/A'):.1f} K")
            print(f"- Max von Mises stress: {derived.get('max_von_mises_stress', 'N/A'):.2e} Pa")
            
            # Field data
            fields = sim.get('fields', {})
            print(f"- Available fields: {list(fields.keys())}")
            
            if 'temperature' in fields:
                temp_field = fields['temperature']
                print(f"- Temperature field shape: {temp_field.shape}")
                print(f"- Temperature range: {np.min(temp_field):.1f} - {np.max(temp_field):.1f} K")
        
    except Exception as e:
        print(f"Could not load Dataset 1: {e}")
        print("Make sure to generate Dataset 1 first using dataset1_generator.py")

def example_2_statistical_analysis():
    """
    Example 2: Statistical analysis of Dataset 1
    """
    print("\n" + "="*60)
    print("Example 2: Statistical Analysis")
    print("="*60)
    
    processor = SOFCDataProcessor()
    
    try:
        # Load multiple simulations for analysis
        simulations = processor.load_dataset1_batch(sim_ids=list(range(10)))
        
        if simulations:
            # Extract features and targets
            features, targets = processor.extract_features_dataset1(simulations)
            
            print(f"Feature matrix shape: {features.shape}")
            print(f"Target matrix shape: {targets.shape}")
            
            # Analyze statistics
            stats = processor.analyze_dataset1_statistics(simulations)
            
            print(f"\nFeature Statistics (first 5 features):")
            feature_names = ['current_density', 'fuel_utilization', 'air_utilization', 
                           'inlet_fuel_temp', 'inlet_air_temp']
            
            for i, name in enumerate(feature_names[:min(5, len(stats['feature_statistics']))]):
                if name in stats['feature_statistics']:
                    stat = stats['feature_statistics'][name]
                    print(f"- {name}: mean={stat['mean']:.3f}, std={stat['std']:.3f}")
            
            print(f"\nTarget Statistics (first 3 targets):")
            target_names = ['cell_voltage', 'max_temperature', 'max_von_mises_stress']
            
            for i, name in enumerate(target_names[:min(3, len(stats['target_statistics']))]):
                if name in stats['target_statistics']:
                    stat = stats['target_statistics'][name]
                    print(f"- {name}: mean={stat['mean']:.3e}, std={stat['std']:.3e}")
            
            # Create visualizations
            print(f"\nCreating visualizations...")
            processor.create_dataset1_visualizations(simulations, "examples/visualizations")
            print(f"Visualizations saved to: examples/visualizations/")
        
    except Exception as e:
        print(f"Statistical analysis failed: {e}")

def example_3_field_visualization():
    """
    Example 3: Visualize 3D field data
    """
    print("\n" + "="*60)
    print("Example 3: Field Visualization")
    print("="*60)
    
    processor = SOFCDataProcessor()
    visualizer = SOFCVisualizer()
    
    try:
        # Load a simulation with field data
        simulation = processor.load_dataset1_simulation(0)
        
        fields = simulation.get('fields', {})
        geometry = simulation.get('metadata', {}).get('geometry', {
            'length': 10e-3, 'width': 10e-3, 'height': 2e-3,
            'nx': 50, 'ny': 30, 'nz': 20
        })
        
        if fields:
            print(f"Available fields: {list(fields.keys())}")
            
            # Visualize temperature field
            if 'temperature' in fields:
                print("Creating temperature field visualization...")
                fig = visualizer.plot_3d_field(
                    fields['temperature'], 'Temperature', geometry,
                    save_path="examples/temperature_field.png"
                )
                plt.close(fig)
                print("Temperature field saved to: examples/temperature_field.png")
            
            # Visualize stress field
            if 'von_mises_stress' in fields:
                print("Creating von Mises stress visualization...")
                fig = visualizer.plot_3d_field(
                    fields['von_mises_stress'], 'Von Mises Stress', geometry,
                    save_path="examples/stress_field.png"
                )
                plt.close(fig)
                print("Stress field saved to: examples/stress_field.png")
            
            # Multi-physics comparison
            print("Creating multi-physics comparison...")
            fig = visualizer.plot_multi_physics_comparison(
                simulation, geometry,
                save_path="examples/multi_physics_comparison.png"
            )
            plt.close(fig)
            print("Multi-physics comparison saved to: examples/multi_physics_comparison.png")
        
    except Exception as e:
        print(f"Field visualization failed: {e}")

def example_4_experimental_data_analysis():
    """
    Example 4: Analyze experimental data (Dataset 2)
    """
    print("\n" + "="*60)
    print("Example 4: Experimental Data Analysis")
    print("="*60)
    
    processor = SOFCDataProcessor()
    visualizer = SOFCVisualizer()
    
    try:
        # Load operational data
        operational_data = processor.load_dataset2_operational()
        
        if not operational_data.empty:
            print(f"Operational data loaded: {len(operational_data)} points")
            print(f"Time range: {operational_data['time_hours'].min():.1f} - {operational_data['time_hours'].max():.1f} hours")
            print(f"Voltage range: {operational_data['voltage'].min():.3f} - {operational_data['voltage'].max():.3f} V")
            
            # Calculate degradation rate
            voltage_trend = np.polyfit(operational_data['time_hours'], operational_data['voltage'], 1)[0]
            print(f"Voltage degradation rate: {voltage_trend*1000:.2f} mV/hour")
        
        # Load experimental data (EIS, thermal, etc.)
        experimental_data = processor.load_dataset2_experimental()
        
        if 'eis' in experimental_data and experimental_data['eis']:
            print(f"EIS measurements loaded: {len(experimental_data['eis'])}")
            
            # Plot EIS evolution
            fig = visualizer.plot_eis_nyquist(
                experimental_data['eis'],
                save_path="examples/eis_evolution.png"
            )
            plt.close(fig)
            print("EIS evolution saved to: examples/eis_evolution.png")
        
        if 'thermal' in experimental_data and experimental_data['thermal']:
            print(f"Thermal images loaded: {len(experimental_data['thermal'])}")
            
            # Plot thermal evolution
            fig = visualizer.plot_thermal_evolution(
                experimental_data['thermal'],
                save_path="examples/thermal_evolution.png"
            )
            plt.close(fig)
            print("Thermal evolution saved to: examples/thermal_evolution.png")
        
        # Load strain data
        strain_data = processor.load_dataset2_strain()
        if not strain_data.empty:
            print(f"Strain data loaded: {len(strain_data)} points")
            
            # Analyze strain evolution
            strain_columns = [col for col in strain_data.columns if 'strain_' in col]
            print(f"Strain measurement locations: {len(strain_columns)}")
            
            for col in strain_columns[:3]:  # Show first 3 locations
                max_strain = strain_data[col].max()
                min_strain = strain_data[col].min()
                print(f"- {col}: {min_strain:.2e} to {max_strain:.2e}")
        
    except Exception as e:
        print(f"Experimental data analysis failed: {e}")

def example_5_realtime_monitoring():
    """
    Example 5: Analyze real-time monitoring data (Dataset 3)
    """
    print("\n" + "="*60)
    print("Example 5: Real-Time Monitoring Data")
    print("="*60)
    
    processor = SOFCDataProcessor()
    visualizer = SOFCVisualizer()
    
    try:
        # Load real-time data
        realtime_data = processor.load_dataset3_realtime()
        
        if 'operational' in realtime_data:
            df = realtime_data['operational']
            print(f"Real-time operational data: {len(df)} points")
            print(f"Simulation duration: {df['simulation_time_hours'].max():.1f} hours")
            
            # Analyze degradation progression
            if 'degradation_state' in df.columns:
                final_degradation = df['degradation_state'].iloc[-1]
                print(f"Final degradation state: {final_degradation:.3f}")
        
        if 'events' in realtime_data:
            events_df = realtime_data['events']
            print(f"Acoustic emission events: {len(events_df)}")
            
            # Analyze event types
            if 'event_type' in events_df.columns:
                event_counts = events_df['event_type'].value_counts()
                print("Event type distribution:")
                for event_type, count in event_counts.items():
                    print(f"- {event_type}: {count}")
        
        if 'eis' in realtime_data and realtime_data['eis']:
            print(f"Real-time EIS measurements: {len(realtime_data['eis'])}")
        
        if 'thermal' in realtime_data and realtime_data['thermal']:
            print(f"Real-time thermal images: {len(realtime_data['thermal'])}")
        
        # Create degradation metrics plot
        if realtime_data:
            fig = visualizer.plot_degradation_metrics(
                realtime_data,
                save_path="examples/degradation_metrics.png"
            )
            plt.close(fig)
            print("Degradation metrics saved to: examples/degradation_metrics.png")
        
    except Exception as e:
        print(f"Real-time data analysis failed: {e}")

def example_6_machine_learning_preparation():
    """
    Example 6: Prepare data for machine learning
    """
    print("\n" + "="*60)
    print("Example 6: Machine Learning Data Preparation")
    print("="*60)
    
    processor = SOFCDataProcessor()
    
    try:
        # Load training data
        simulations = processor.load_dataset1_batch(sim_ids=list(range(20)))
        
        if simulations:
            # Extract features and targets
            features, targets = processor.extract_features_dataset1(simulations)
            
            print(f"Training data shape:")
            print(f"- Features: {features.shape}")
            print(f"- Targets: {targets.shape}")
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            feature_scaler = StandardScaler()
            target_scaler = StandardScaler()
            
            features_scaled = feature_scaler.fit_transform(features)
            targets_scaled = target_scaler.fit_transform(targets)
            
            print(f"\nNormalized data statistics:")
            print(f"- Features mean: {np.mean(features_scaled, axis=0)[:3]}")
            print(f"- Features std: {np.std(features_scaled, axis=0)[:3]}")
            
            # Split into train/validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                features_scaled, targets_scaled, test_size=0.2, random_state=42
            )
            
            print(f"\nTrain/validation split:")
            print(f"- Training samples: {X_train.shape[0]}")
            print(f"- Validation samples: {X_val.shape[0]}")
            
            # Correlation analysis
            import pandas as pd
            
            # Create feature names
            feature_names = [
                'current_density', 'fuel_utilization', 'air_utilization',
                'inlet_fuel_temp', 'inlet_air_temp', 'h2_percentage',
                'h2o_percentage', 'co_percentage', 'ch4_percentage',
                'electrode_porosity', 'electrode_tortuosity', 'anode_conductivity',
                'cathode_conductivity', 'electrolyte_thickness', 'electrode_thickness',
                'initial_crack_length', 'porosity_degradation'
            ]
            
            # Find most important features (correlation with first target - voltage)
            correlations = []
            for i in range(features.shape[1]):
                corr = np.corrcoef(features[:, i], targets[:, 0])[0, 1]
                correlations.append((feature_names[i] if i < len(feature_names) else f'feature_{i}', abs(corr)))
            
            # Sort by correlation strength
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 5 features correlated with cell voltage:")
            for i, (name, corr) in enumerate(correlations[:5]):
                print(f"{i+1}. {name}: {corr:.3f}")
            
            # Save processed data
            np.savez('examples/ml_data.npz',
                    X_train=X_train, X_val=X_val,
                    y_train=y_train, y_val=y_val,
                    feature_names=feature_names[:features.shape[1]])
            
            print(f"\nProcessed ML data saved to: examples/ml_data.npz")
        
    except Exception as e:
        print(f"ML data preparation failed: {e}")

def main():
    """
    Run all examples
    """
    print("SOFC Digital Twin Dataset - Basic Usage Examples")
    print("This script demonstrates how to load, process, and analyze the datasets")
    print("\nNote: Make sure to generate the datasets first using the respective generators")
    
    # Create examples directory
    os.makedirs("examples", exist_ok=True)
    os.makedirs("examples/visualizations", exist_ok=True)
    
    # Run examples
    example_1_load_and_explore_dataset1()
    example_2_statistical_analysis()
    example_3_field_visualization()
    example_4_experimental_data_analysis()
    example_5_realtime_monitoring()
    example_6_machine_learning_preparation()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Check the 'examples/' directory for generated files and visualizations")
    print("="*60)

if __name__ == "__main__":
    main()