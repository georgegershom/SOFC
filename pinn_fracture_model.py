#!/usr/bin/env python3
"""
Physics-Informed Neural Network (PINN) for SOFC Fracture Prediction
==================================================================

This module implements a PINN for predicting fracture evolution in SOFC electrolytes
using the generated ground truth dataset.

The PINN incorporates:
1. Phase-field fracture mechanics equations
2. Thermomechanical coupling
3. Material property constraints
4. Boundary conditions from SOFC geometry

Author: AI Assistant
Date: 2025-10-09
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import time

class SOFCFracturePINN:
    """
    Physics-Informed Neural Network for SOFC fracture prediction.
    """
    
    def __init__(self, 
                 layers: List[int] = [4, 64, 64, 64, 64, 1],
                 activation: str = 'tanh',
                 learning_rate: float = 1e-3):
        """
        Initialize the PINN model.
        
        Args:
            layers: List defining network architecture [input_dim, hidden1, ..., output_dim]
            activation: Activation function ('tanh', 'relu', 'swish')
            learning_rate: Learning rate for optimization
        """
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Physical parameters (from YSZ properties)
        self.material_params = {
            'E': 170e9,  # Young's modulus (Pa)
            'nu': 0.23,  # Poisson's ratio
            'alpha': 10.5e-6,  # Thermal expansion coefficient (1/K)
            'Gc': 2.7e-3,  # Critical energy release rate (J/m²)
            'l0': 2.34e-6,  # Length scale parameter (m)
            'kappa': 1e-6,  # Mobility parameter
        }
        
        # Normalization parameters
        self.norm_params = {
            'x_scale': 150e-6,  # Characteristic length (electrolyte thickness)
            't_scale': 3600,    # Characteristic time (1 hour)
            'phi_scale': 1.0,   # Phase field scale
            'stress_scale': 165e6,  # Characteristic stress (material strength)
        }
        
        # Build the neural network
        self.model = self._build_network()
        
        # Compile with custom optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
    def _build_network(self) -> keras.Model:
        """Build the neural network architecture."""
        inputs = keras.Input(shape=(self.layers[0],), name='input')
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(self.layers[1:-1]):
            x = keras.layers.Dense(
                units, 
                activation=self.activation,
                kernel_initializer='glorot_normal',
                name=f'hidden_{i+1}'
            )(x)
            
        # Output layer (phase field)
        outputs = keras.layers.Dense(
            self.layers[-1], 
            activation='sigmoid',  # Phase field bounded [0,1]
            name='phase_field'
        )(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='SOFC_Fracture_PINN')
        return model
    
    def physics_loss(self, x_batch: tf.Tensor, phi_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute physics-based loss from Allen-Cahn equation and fracture mechanics.
        
        Args:
            x_batch: Input coordinates [x, y, z, t]
            phi_pred: Predicted phase field values
            
        Returns:
            Physics loss tensor
        """
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x_batch)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x_batch)
                phi = self.model(x_batch)
            
            # First-order derivatives
            phi_x = tape1.gradient(phi, x_batch)[:, 0:1]
            phi_y = tape1.gradient(phi, x_batch)[:, 1:2]
            phi_z = tape1.gradient(phi, x_batch)[:, 2:3]
            phi_t = tape1.gradient(phi, x_batch)[:, 3:4]
            
        # Second-order derivatives (Laplacian)
        phi_xx = tape2.gradient(phi_x, x_batch)[:, 0:1]
        phi_yy = tape2.gradient(phi_y, x_batch)[:, 1:2]
        phi_zz = tape2.gradient(phi_z, x_batch)[:, 2:3]
        laplacian = phi_xx + phi_yy + phi_zz
        
        # Gradient magnitude squared
        grad_phi_sq = phi_x**2 + phi_y**2 + phi_z**2
        
        # Allen-Cahn equation for phase field evolution
        # ∂φ/∂t = κ[Gc*l0*∇²φ - (Gc/l0)*φ(1-φ)(1-2φ) + driving_force]
        
        # Thermodynamic driving force (simplified stress-based)
        stress_field = self._compute_stress_field(x_batch)
        driving_force = stress_field * (1 - phi)**2
        
        # Allen-Cahn residual
        ac_residual = (phi_t - self.material_params['kappa'] * (
            self.material_params['Gc'] * self.material_params['l0'] * laplacian -
            (self.material_params['Gc'] / self.material_params['l0']) * 
            phi * (1 - phi) * (1 - 2*phi) +
            driving_force
        ))
        
        # Gradient regularization (prevents sharp interfaces)
        grad_regularization = tf.reduce_mean(grad_phi_sq)
        
        # Physics loss
        physics_loss = tf.reduce_mean(tf.square(ac_residual)) + 0.1 * grad_regularization
        
        return physics_loss
    
    def _compute_stress_field(self, x_batch: tf.Tensor) -> tf.Tensor:
        """
        Compute simplified stress field for driving force.
        
        Args:
            x_batch: Spatial coordinates [x, y, z, t]
            
        Returns:
            Normalized stress field
        """
        x, y, z, t = x_batch[:, 0:1], x_batch[:, 1:2], x_batch[:, 2:3], x_batch[:, 3:4]
        
        # Thermal stress (uniform)
        thermal_stress = 1.0  # Normalized
        
        # Edge stress concentrations
        edge_factor = 1.0 + 0.5 * tf.exp(-tf.minimum(
            tf.minimum(x, 1-x), tf.minimum(y, 1-y)
        ) / 0.1)
        
        # Time-dependent loading (thermal cycling)
        time_factor = 1.0 + 0.2 * tf.sin(0.1 * t)
        
        stress_field = thermal_stress * edge_factor * time_factor
        
        return stress_field
    
    def data_loss(self, phi_pred: tf.Tensor, phi_true: tf.Tensor) -> tf.Tensor:
        """
        Compute data fitting loss.
        
        Args:
            phi_pred: Predicted phase field
            phi_true: True phase field from dataset
            
        Returns:
            Data loss tensor
        """
        return tf.reduce_mean(tf.square(phi_pred - phi_true))
    
    def boundary_loss(self, x_boundary: tf.Tensor) -> tf.Tensor:
        """
        Compute boundary condition loss.
        
        Args:
            x_boundary: Boundary coordinates
            
        Returns:
            Boundary loss tensor
        """
        phi_boundary = self.model(x_boundary)
        
        # No-flux boundary conditions (∇φ·n = 0)
        # For simplicity, enforce φ = 0 at boundaries (intact material)
        boundary_loss = tf.reduce_mean(tf.square(phi_boundary))
        
        return boundary_loss
    
    @tf.function
    def train_step(self, 
                   x_physics: tf.Tensor, 
                   x_data: tf.Tensor, 
                   phi_data: tf.Tensor,
                   x_boundary: tf.Tensor,
                   lambda_physics: float = 1.0,
                   lambda_data: float = 10.0,
                   lambda_boundary: float = 1.0) -> Dict[str, tf.Tensor]:
        """
        Perform one training step.
        
        Args:
            x_physics: Physics collocation points
            x_data: Data points coordinates
            phi_data: True phase field values at data points
            x_boundary: Boundary points
            lambda_physics: Physics loss weight
            lambda_data: Data loss weight
            lambda_boundary: Boundary loss weight
            
        Returns:
            Dictionary of losses
        """
        with tf.GradientTape() as tape:
            # Predictions
            phi_physics = self.model(x_physics)
            phi_pred = self.model(x_data)
            
            # Compute losses
            loss_physics = self.physics_loss(x_physics, phi_physics)
            loss_data = self.data_loss(phi_pred, phi_data)
            loss_boundary = self.boundary_loss(x_boundary)
            
            # Total loss
            total_loss = (lambda_physics * loss_physics + 
                         lambda_data * loss_data + 
                         lambda_boundary * loss_boundary)
        
        # Compute gradients and update
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return {
            'total_loss': total_loss,
            'physics_loss': loss_physics,
            'data_loss': loss_data,
            'boundary_loss': loss_boundary
        }
    
    def generate_training_points(self, 
                               n_physics: int = 5000,
                               n_boundary: int = 1000) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generate collocation points for physics and boundary losses.
        
        Args:
            n_physics: Number of physics collocation points
            n_boundary: Number of boundary points
            
        Returns:
            Physics points and boundary points
        """
        # Physics points (interior domain)
        x_physics = tf.random.uniform((n_physics, 4), minval=0.0, maxval=1.0)
        
        # Boundary points (domain boundaries)
        n_per_face = n_boundary // 6
        
        # x=0 and x=1 faces
        x_bound_1 = tf.concat([
            tf.zeros((n_per_face, 1)),
            tf.random.uniform((n_per_face, 3))
        ], axis=1)
        x_bound_2 = tf.concat([
            tf.ones((n_per_face, 1)),
            tf.random.uniform((n_per_face, 3))
        ], axis=1)
        
        # y=0 and y=1 faces
        x_bound_3 = tf.concat([
            tf.random.uniform((n_per_face, 1)),
            tf.zeros((n_per_face, 1)),
            tf.random.uniform((n_per_face, 2))
        ], axis=1)
        x_bound_4 = tf.concat([
            tf.random.uniform((n_per_face, 1)),
            tf.ones((n_per_face, 1)),
            tf.random.uniform((n_per_face, 2))
        ], axis=1)
        
        # z=0 and z=1 faces
        x_bound_5 = tf.concat([
            tf.random.uniform((n_per_face, 2)),
            tf.zeros((n_per_face, 1)),
            tf.random.uniform((n_per_face, 1))
        ], axis=1)
        x_bound_6 = tf.concat([
            tf.random.uniform((n_per_face, 2)),
            tf.ones((n_per_face, 1)),
            tf.random.uniform((n_per_face, 1))
        ], axis=1)
        
        x_boundary = tf.concat([
            x_bound_1, x_bound_2, x_bound_3, x_bound_4, x_bound_5, x_bound_6
        ], axis=0)
        
        return x_physics, x_boundary
    
    def load_training_data(self, dataset_path: str, sample_ids: List[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load training data from the fracture dataset.
        
        Args:
            dataset_path: Path to the fracture dataset
            sample_ids: List of sample IDs to use (None for all)
            
        Returns:
            Coordinates and phase field values
        """
        dataset_dir = Path(dataset_path)
        
        if sample_ids is None:
            sample_dirs = sorted(dataset_dir.glob('sample_*'))
            sample_ids = [int(d.name.split('_')[1]) for d in sample_dirs]
        
        all_coords = []
        all_phi = []
        
        for sample_id in sample_ids:
            sample_dir = dataset_dir / f'sample_{sample_id:03d}'
            
            # Load phase field data
            with h5py.File(sample_dir / 'phase_field_data.h5', 'r') as f:
                phase_field = f['phase_field'][:]
                time_array = f['physical_time'][:]
            
            # Create coordinate grid
            nx, ny, nz, nt = phase_field.shape
            x = np.linspace(0, 1, nx)
            y = np.linspace(0, 1, ny)
            z = np.linspace(0, 1, nz)
            t = time_array / time_array[-1]  # Normalize time
            
            X, Y, Z, T = np.meshgrid(x, y, z, t, indexing='ij')
            
            # Flatten arrays
            coords = np.stack([X.ravel(), Y.ravel(), Z.ravel(), T.ravel()], axis=1)
            phi_values = phase_field.ravel()
            
            all_coords.append(coords)
            all_phi.append(phi_values)
        
        # Concatenate all samples
        coords_combined = np.concatenate(all_coords, axis=0)
        phi_combined = np.concatenate(all_phi, axis=0)
        
        # Subsample for computational efficiency
        n_total = len(coords_combined)
        n_sample = min(50000, n_total)  # Use up to 50k points
        indices = np.random.choice(n_total, n_sample, replace=False)
        
        coords_sampled = coords_combined[indices]
        phi_sampled = phi_combined[indices]
        
        return tf.constant(coords_sampled, dtype=tf.float32), tf.constant(phi_sampled, dtype=tf.float32)
    
    def train(self, 
              dataset_path: str,
              epochs: int = 1000,
              sample_ids: List[int] = None,
              save_path: str = 'sofc_pinn_model.h5') -> Dict[str, List[float]]:
        """
        Train the PINN model.
        
        Args:
            dataset_path: Path to fracture dataset
            epochs: Number of training epochs
            sample_ids: Sample IDs to use for training
            save_path: Path to save trained model
            
        Returns:
            Training history
        """
        print("Loading training data...")
        x_data, phi_data = self.load_training_data(dataset_path, sample_ids)
        
        print("Generating collocation points...")
        x_physics, x_boundary = self.generate_training_points()
        
        # Training history
        history = {
            'total_loss': [],
            'physics_loss': [],
            'data_loss': [],
            'boundary_loss': []
        }
        
        print(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Perform training step
            losses = self.train_step(x_physics, x_data, phi_data, x_boundary)
            
            # Record losses
            for key in history.keys():
                history[key].append(float(losses[key]))
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Total Loss: {losses['total_loss']:.6f} - "
                      f"Physics: {losses['physics_loss']:.6f} - "
                      f"Data: {losses['data_loss']:.6f} - "
                      f"Boundary: {losses['boundary_loss']:.6f} - "
                      f"Time: {elapsed:.1f}s")
        
        # Save model
        self.model.save_weights(save_path)
        print(f"Model saved to {save_path}")
        
        return history
    
    def predict_fracture_evolution(self, 
                                 x_coords: np.ndarray, 
                                 y_coords: np.ndarray,
                                 z_coords: np.ndarray,
                                 time_points: np.ndarray) -> np.ndarray:
        """
        Predict fracture evolution at specified coordinates and times.
        
        Args:
            x_coords: X coordinates (normalized [0,1])
            y_coords: Y coordinates (normalized [0,1])
            z_coords: Z coordinates (normalized [0,1])
            time_points: Time points (normalized [0,1])
            
        Returns:
            Predicted phase field evolution
        """
        # Create coordinate grid
        X, Y, Z, T = np.meshgrid(x_coords, y_coords, z_coords, time_points, indexing='ij')
        coords = np.stack([X.ravel(), Y.ravel(), Z.ravel(), T.ravel()], axis=1)
        
        # Predict
        coords_tf = tf.constant(coords, dtype=tf.float32)
        phi_pred = self.model(coords_tf).numpy()
        
        # Reshape to grid
        phi_evolution = phi_pred.reshape(X.shape)
        
        return phi_evolution
    
    def plot_training_history(self, history: Dict[str, List[float]]):
        """Plot training loss history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('PINN Training History', fontsize=16)
        
        # Total loss
        axes[0, 0].plot(history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True)
        
        # Physics loss
        axes[0, 1].plot(history['physics_loss'], color='orange')
        axes[0, 1].set_title('Physics Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Data loss
        axes[1, 0].plot(history['data_loss'], color='green')
        axes[1, 0].set_title('Data Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Boundary loss
        axes[1, 1].plot(history['boundary_loss'], color='red')
        axes[1, 1].set_title('Boundary Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function to demonstrate PINN training."""
    print("SOFC Fracture PINN Training")
    print("=" * 40)
    
    # Initialize PINN
    pinn = SOFCFracturePINN(
        layers=[4, 64, 64, 64, 1],
        activation='tanh',
        learning_rate=1e-3
    )
    
    print(f"Model architecture: {pinn.layers}")
    print(f"Total parameters: {pinn.model.count_params()}")
    
    # Train the model
    history = pinn.train(
        dataset_path='fracture_dataset',
        epochs=500,  # Reduced for demonstration
        sample_ids=[0, 1, 2]  # Use first 3 samples
    )
    
    # Plot training history
    pinn.plot_training_history(history)
    
    # Demonstrate prediction
    print("\\nGenerating predictions...")
    x = np.linspace(0, 1, 32)
    y = np.linspace(0, 1, 32)
    z = np.array([0.5])  # Middle z-slice
    t = np.linspace(0, 1, 10)
    
    phi_pred = pinn.predict_fracture_evolution(x, y, z, t)
    
    # Visualize prediction
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('PINN Fracture Prediction', fontsize=16)
    
    for i in range(10):
        ax = axes[i//5, i%5]
        im = ax.imshow(phi_pred[:, :, 0, i], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f't = {t[i]:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Phase field')
    
    plt.tight_layout()
    plt.show()
    
    print("PINN training and prediction complete!")


if __name__ == '__main__':
    main()