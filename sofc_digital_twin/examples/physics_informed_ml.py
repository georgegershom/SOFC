"""
Physics-Informed Machine Learning Example
Demonstrates how to train physics-informed neural networks using the SOFC dataset
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from utils.data_processor import SOFCDataProcessor

class PhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network for SOFC modeling
    """
    
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 128, 64]):
        super(PhysicsInformedNN, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Physics constants
        self.F = 96485.0  # Faraday constant
        self.R = 8.314    # Gas constant
        
    def forward(self, x):
        return self.network(x)
    
    def physics_loss(self, inputs, outputs, targets):
        """
        Compute physics-based loss terms
        """
        # Extract inputs (operating conditions)
        current_density = inputs[:, 0]  # A/cm²
        fuel_util = inputs[:, 1]        # Fuel utilization
        temp_fuel = inputs[:, 3]        # Fuel temperature
        
        # Extract predictions
        voltage_pred = outputs[:, 0]    # Cell voltage
        temp_max_pred = outputs[:, 1]   # Max temperature
        
        # Physics constraint 1: Nernst equation (simplified)
        # E_nernst = E0 + (R*T)/(n*F) * ln(reactant_activity)
        T_kelvin = temp_fuel + 273.15
        E_nernst = 1.2 + (self.R * T_kelvin) / (2 * self.F) * torch.log(fuel_util + 1e-6)
        
        # Voltage should be less than Nernst potential
        nernst_loss = torch.mean(torch.relu(voltage_pred - E_nernst))
        
        # Physics constraint 2: Heat generation
        # Heat generation should increase with current density and decrease with voltage
        heat_generation = current_density * (E_nernst - voltage_pred)
        temp_physics_loss = torch.mean(torch.relu(temp_max_pred - temp_fuel - heat_generation * 100))
        
        # Physics constraint 3: Current-voltage relationship
        # Higher current should generally lead to lower voltage (polarization)
        cv_correlation = torch.corrcoef(torch.stack([current_density, voltage_pred]))[0, 1]
        cv_loss = torch.relu(cv_correlation + 0.5)  # Should be negative correlation
        
        return nernst_loss + temp_physics_loss + cv_loss

class SOFCTrainer:
    """
    Trainer for physics-informed SOFC models
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        
        # Loss weights
        self.data_loss_weight = 1.0
        self.physics_loss_weight = 0.1
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.physics_losses = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_physics_loss = 0
        
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            outputs = self.model(batch_inputs)
            
            # Data loss (MSE)
            data_loss = nn.MSELoss()(outputs, batch_targets)
            
            # Physics loss
            physics_loss = self.model.physics_loss(batch_inputs, outputs, batch_targets)
            
            # Total loss
            total_batch_loss = (self.data_loss_weight * data_loss + 
                              self.physics_loss_weight * physics_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_physics_loss += physics_loss.item()
        
        return total_loss / len(train_loader), total_physics_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_inputs)
                loss = nn.MSELoss()(outputs, batch_targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=100):
        """Full training loop"""
        print(f"Training physics-informed neural network for {epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, physics_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.physics_losses.append(physics_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'examples/best_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, "
                      f"Val Loss={val_loss:.6f}, Physics Loss={physics_loss:.6f}")
            
            # Early stopping
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('examples/best_model.pth'))
    
    def plot_training_history(self, save_path='examples/training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        axes[0].plot(self.train_losses, label='Training Loss', alpha=0.7)
        axes[0].plot(self.val_losses, label='Validation Loss', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Physics loss
        axes[1].plot(self.physics_losses, label='Physics Loss', color='red', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Physics Loss')
        axes[1].set_title('Physics Constraint Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history saved to: {save_path}")

def evaluate_model(model, X_test, y_test, feature_scaler, target_scaler):
    """
    Evaluate trained model performance
    """
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_pred_scaled = model(X_test_tensor).numpy()
    
    # Inverse transform predictions
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_test_orig = target_scaler.inverse_transform(y_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)
    
    print(f"\nModel Performance:")
    print(f"- MSE: {mse:.6f}")
    print(f"- R² Score: {r2:.4f}")
    
    # Per-target metrics
    target_names = ['Cell Voltage', 'Max Temperature', 'Max Stress']
    
    for i, name in enumerate(target_names[:y_test_orig.shape[1]]):
        mse_target = mean_squared_error(y_test_orig[:, i], y_pred[:, i])
        r2_target = r2_score(y_test_orig[:, i], y_pred[:, i])
        print(f"- {name}: MSE={mse_target:.6f}, R²={r2_target:.4f}")
    
    return y_pred, y_test_orig

def plot_predictions(y_pred, y_test, save_path='examples/predictions.png'):
    """
    Plot prediction vs actual values
    """
    target_names = ['Cell Voltage (V)', 'Max Temperature (K)', 'Max Stress (Pa)']
    n_targets = min(len(target_names), y_test.shape[1])
    
    fig, axes = plt.subplots(1, n_targets, figsize=(5*n_targets, 5))
    if n_targets == 1:
        axes = [axes]
    
    for i in range(n_targets):
        axes[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(y_test[:, i].min(), y_pred[:, i].min())
        max_val = max(y_test[:, i].max(), y_pred[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        axes[i].set_xlabel(f'Actual {target_names[i]}')
        axes[i].set_ylabel(f'Predicted {target_names[i]}')
        axes[i].set_title(f'{target_names[i]} Predictions')
        axes[i].grid(True, alpha=0.3)
        
        # R² annotation
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Prediction plots saved to: {save_path}")

def main():
    """
    Main function demonstrating physics-informed ML training
    """
    print("="*60)
    print("Physics-Informed Machine Learning for SOFC Digital Twin")
    print("="*60)
    
    # Create examples directory
    os.makedirs("examples", exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    processor = SOFCDataProcessor()
    
    try:
        # Load simulations
        simulations = processor.load_dataset1_batch(sim_ids=list(range(50)))
        
        if not simulations:
            print("No simulation data found. Please generate Dataset 1 first.")
            return
        
        # Extract features and targets
        features, targets = processor.extract_features_dataset1(simulations)
        
        print(f"Dataset loaded:")
        print(f"- Samples: {features.shape[0]}")
        print(f"- Features: {features.shape[1]}")
        print(f"- Targets: {targets.shape[1]}")
        
        # Normalize data
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        features_scaled = feature_scaler.fit_transform(features)
        targets_scaled = target_scaler.fit_transform(targets)
        
        # Train/validation/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            features_scaled, targets_scaled, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )
        
        print(f"Data split:")
        print(f"- Training: {X_train.shape[0]} samples")
        print(f"- Validation: {X_val.shape[0]} samples")
        print(f"- Test: {X_test.shape[0]} samples")
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        print("\nInitializing physics-informed neural network...")
        input_dim = features.shape[1]
        output_dim = targets.shape[1]
        
        model = PhysicsInformedNN(input_dim, output_dim, hidden_dims=[64, 128, 128, 64])
        
        print(f"Model architecture:")
        print(f"- Input dimension: {input_dim}")
        print(f"- Output dimension: {output_dim}")
        print(f"- Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        print("\nTraining model...")
        trainer = SOFCTrainer(model)
        trainer.train(train_loader, val_loader, epochs=100)
        
        # Plot training history
        trainer.plot_training_history()
        
        # Evaluate model
        print("\nEvaluating model...")
        y_pred, y_test_orig = evaluate_model(model, X_test, y_test, feature_scaler, target_scaler)
        
        # Plot predictions
        plot_predictions(y_pred, y_test_orig)
        
        # Feature importance analysis
        print("\nAnalyzing feature importance...")
        
        # Simple sensitivity analysis
        model.eval()
        baseline_input = torch.FloatTensor(X_test[:100].mean(axis=0, keepdims=True))
        baseline_output = model(baseline_input).detach().numpy()
        
        feature_names = [
            'Current Density', 'Fuel Utilization', 'Air Utilization',
            'Inlet Fuel Temp', 'Inlet Air Temp', 'H₂ %', 'H₂O %', 'CO %', 'CH₄ %',
            'Electrode Porosity', 'Electrode Tortuosity', 'Anode Conductivity',
            'Cathode Conductivity', 'Electrolyte Thickness', 'Electrode Thickness',
            'Initial Crack Length', 'Porosity Degradation'
        ]
        
        sensitivities = []
        perturbation = 0.1  # 10% perturbation
        
        for i in range(input_dim):
            # Perturb feature
            perturbed_input = baseline_input.clone()
            perturbed_input[0, i] += perturbation
            
            # Get output
            perturbed_output = model(perturbed_input).detach().numpy()
            
            # Calculate sensitivity (change in voltage)
            sensitivity = abs(perturbed_output[0, 0] - baseline_output[0, 0]) / perturbation
            sensitivities.append(sensitivity)
        
        # Sort by sensitivity
        sensitivity_pairs = list(zip(feature_names[:input_dim], sensitivities))
        sensitivity_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 5 most sensitive features (for cell voltage):")
        for i, (name, sens) in enumerate(sensitivity_pairs[:5]):
            print(f"{i+1}. {name}: {sens:.6f}")
        
        # Save model and scalers
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'input_dim': input_dim,
            'output_dim': output_dim
        }, 'examples/sofc_model.pth')
        
        print(f"\nModel saved to: examples/sofc_model.pth")
        
        print("\n" + "="*60)
        print("Physics-informed ML training completed successfully!")
        print("Check the 'examples/' directory for results and visualizations")
        print("="*60)
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()