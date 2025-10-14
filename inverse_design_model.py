"""
Inverse Design Model for Welding Parameter Optimization
Using Variational Autoencoder (VAE) and Conditional GAN approaches
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
import json
import os


class WeldingDataset(Dataset):
    """PyTorch Dataset for welding data"""
    
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for Inverse Design
    Takes desired outputs (Y) as condition and generates inputs (X)
    """
    
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int = 32):
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        # Encoder: (X, Y) -> latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + output_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder: (latent, Y) -> X
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + output_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim)
        )
        
    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input and condition to latent space"""
        xy = torch.cat([x, y], dim=1)
        h = self.encoder(xy)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Decode latent code and condition to input parameters"""
        zy = torch.cat([z, y], dim=1)
        return self.decoder(zy)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE"""
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar
    
    def generate(self, y: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Generate input parameters for desired outputs"""
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(n_samples, self.latent_dim)
            if y.dim() == 1:
                y = y.unsqueeze(0).repeat(n_samples, 1)
            # Decode to get input parameters
            x_generated = self.decode(z, y)
        return x_generated


class InverseDesignGAN(nn.Module):
    """
    Conditional GAN for Inverse Design
    Generator: Y -> X
    Discriminator: (X, Y) -> Real/Fake
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super(InverseDesignGAN, self).__init__()
        
        # Generator: Y -> X
        self.generator = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim),
            nn.Tanh()  # Assuming normalized inputs
        )
        
        # Discriminator: (X, Y) -> Real/Fake
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim + output_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def generate(self, y: torch.Tensor) -> torch.Tensor:
        """Generate input parameters for desired outputs"""
        return self.generator(y)
    
    def discriminate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Discriminate real/fake pairs"""
        xy = torch.cat([x, y], dim=1)
        return self.discriminator(xy)


class InverseDesignTrainer:
    """
    Trainer for inverse design models
    """
    
    def __init__(self, model_type: str = 'vae'):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def load_data(self, data_path: str):
        """Load ML-ready data"""
        self.X_train = np.load(f'{data_path}/X_train.npy')
        self.Y_train = np.load(f'{data_path}/Y_train.npy')
        self.X_val = np.load(f'{data_path}/X_val.npy')
        self.Y_val = np.load(f'{data_path}/Y_val.npy')
        self.X_test = np.load(f'{data_path}/X_test.npy')
        self.Y_test = np.load(f'{data_path}/Y_test.npy')
        
        # Load feature names
        with open(f'{data_path}/feature_names.json', 'r') as f:
            feature_names = json.load(f)
            self.input_features = feature_names['input_features']
            self.output_features = feature_names['output_features']
        
        # Normalize data
        self.X_train = self.scaler_x.fit_transform(self.X_train)
        self.Y_train = self.scaler_y.fit_transform(self.Y_train)
        self.X_val = self.scaler_x.transform(self.X_val)
        self.Y_val = self.scaler_y.transform(self.Y_val)
        self.X_test = self.scaler_x.transform(self.X_test)
        self.Y_test = self.scaler_y.transform(self.Y_test)
        
        print(f"Data loaded: {len(self.X_train)} training samples")
        
    def train_vae(self, epochs: int = 100, batch_size: int = 64, lr: float = 1e-3):
        """Train Conditional VAE model"""
        
        # Create data loaders
        train_dataset = WeldingDataset(self.X_train, self.Y_train)
        val_dataset = WeldingDataset(self.X_val, self.Y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_dim = self.X_train.shape[1]
        output_dim = self.Y_train.shape[1]
        self.model = ConditionalVAE(input_dim, output_dim).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': []}
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_recon = 0
            train_kl = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                x_recon, mu, logvar = self.model(batch_x, batch_y)
                
                # Calculate losses
                recon_loss = nn.MSELoss()(x_recon, batch_x)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_x.size(0)
                loss = recon_loss + 0.1 * kl_loss  # Beta-VAE weight
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_recon += recon_loss.item()
                train_kl += kl_loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    x_recon, mu, logvar = self.model(batch_x, batch_y)
                    
                    recon_loss = nn.MSELoss()(x_recon, batch_x)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_x.size(0)
                    loss = recon_loss + 0.1 * kl_loss
                    
                    val_loss += loss.item()
            
            # Record history
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['recon_loss'].append(train_recon / len(train_loader))
            history['kl_loss'].append(train_kl / len(train_loader))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return history
    
    def train_gan(self, epochs: int = 200, batch_size: int = 64, lr: float = 2e-4):
        """Train Conditional GAN model"""
        
        # Create data loaders
        train_dataset = WeldingDataset(self.X_train, self.Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize models
        input_dim = self.X_train.shape[1]
        output_dim = self.Y_train.shape[1]
        self.gan_model = InverseDesignGAN(input_dim, output_dim).to(self.device)
        
        # Optimizers
        g_optimizer = optim.Adam(self.gan_model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(self.gan_model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Training history
        history = {'g_loss': [], 'd_loss': [], 'd_real': [], 'd_fake': []}
        
        # Training loop
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            d_real_scores = []
            d_fake_scores = []
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_size = batch_x.size(0)
                
                # Labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                # Real data
                real_output = self.gan_model.discriminate(batch_x, batch_y)
                d_real_loss = criterion(real_output, real_labels)
                
                # Fake data
                fake_x = self.gan_model.generate(batch_y)
                fake_output = self.gan_model.discriminate(fake_x.detach(), batch_y)
                d_fake_loss = criterion(fake_output, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                fake_x = self.gan_model.generate(batch_y)
                fake_output = self.gan_model.discriminate(fake_x, batch_y)
                g_loss = criterion(fake_output, real_labels)
                
                g_loss.backward()
                g_optimizer.step()
                
                # Record metrics
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                d_real_scores.append(real_output.mean().item())
                d_fake_scores.append(fake_output.mean().item())
            
            # Record history
            history['g_loss'].append(np.mean(g_losses))
            history['d_loss'].append(np.mean(d_losses))
            history['d_real'].append(np.mean(d_real_scores))
            history['d_fake'].append(np.mean(d_fake_scores))
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - G Loss: {np.mean(g_losses):.4f}, "
                      f"D Loss: {np.mean(d_losses):.4f}, "
                      f"D(real): {np.mean(d_real_scores):.3f}, "
                      f"D(fake): {np.mean(d_fake_scores):.3f}")
        
        return history
    
    def inverse_design(self, desired_outputs: Dict[str, float], n_solutions: int = 10) -> pd.DataFrame:
        """
        Generate welding parameters for desired outputs
        
        Parameters:
        -----------
        desired_outputs: Dictionary of desired output values
            e.g., {'tensile_strength': 3500, 'contact_resistance': 15}
        n_solutions: Number of solutions to generate
        
        Returns:
        --------
        DataFrame with generated input parameters
        """
        
        # Prepare desired outputs vector
        y_desired = np.zeros(len(self.output_features))
        for feature, value in desired_outputs.items():
            if feature in self.output_features:
                idx = self.output_features.index(feature)
                # Normalize the value
                y_desired[idx] = (value - self.scaler_y.mean_[idx]) / self.scaler_y.scale_[idx]
        
        y_tensor = torch.FloatTensor(y_desired).to(self.device)
        
        # Generate solutions
        if self.model_type == 'vae':
            x_generated = self.model.generate(y_tensor, n_solutions)
        else:  # GAN
            y_batch = y_tensor.unsqueeze(0).repeat(n_solutions, 1)
            x_generated = self.gan_model.generate(y_batch)
        
        # Convert back to original scale
        x_generated_np = x_generated.cpu().detach().numpy()
        x_original = self.scaler_x.inverse_transform(x_generated_np)
        
        # Create DataFrame
        solutions_df = pd.DataFrame(x_original, columns=self.input_features)
        
        # Add solution ranking based on feasibility
        solutions_df['feasibility_score'] = self._calculate_feasibility(solutions_df)
        solutions_df = solutions_df.sort_values('feasibility_score', ascending=False)
        
        return solutions_df
    
    def _calculate_feasibility(self, solutions: pd.DataFrame) -> np.ndarray:
        """Calculate feasibility score for generated solutions"""
        scores = np.ones(len(solutions))
        
        # Check physical constraints
        for idx in range(len(solutions)):
            # Power constraints
            if solutions.iloc[idx]['laser_power'] < 300 or solutions.iloc[idx]['laser_power'] > 5000:
                scores[idx] *= 0.5
            
            # Speed constraints
            if solutions.iloc[idx]['welding_speed'] < 5 or solutions.iloc[idx]['welding_speed'] > 200:
                scores[idx] *= 0.5
            
            # Heat input constraints
            heat_input = solutions.iloc[idx]['laser_power'] / solutions.iloc[idx]['welding_speed'] / 1000
            if heat_input < 0.01 or heat_input > 1.0:
                scores[idx] *= 0.7
            
            # Beam size constraints
            if 'beam_spot_size' in solutions.columns:
                if solutions.iloc[idx]['beam_spot_size'] < 50 or solutions.iloc[idx]['beam_spot_size'] > 1000:
                    scores[idx] *= 0.8
        
        return scores
    
    def save_model(self, path: str = 'models'):
        """Save trained model"""
        os.makedirs(path, exist_ok=True)
        
        if self.model_type == 'vae':
            torch.save(self.model.state_dict(), f'{path}/vae_model.pth')
        else:
            torch.save(self.gan_model.state_dict(), f'{path}/gan_model.pth')
        
        # Save scalers
        import joblib
        joblib.dump(self.scaler_x, f'{path}/scaler_x.pkl')
        joblib.dump(self.scaler_y, f'{path}/scaler_y.pkl')
        
        # Save feature names
        with open(f'{path}/features.json', 'w') as f:
            json.dump({
                'input_features': self.input_features,
                'output_features': self.output_features
            }, f, indent=2)
        
        print(f"Model saved to {path}/")
    
    def visualize_training(self, history: Dict):
        """Visualize training history"""
        fig, axes = plt.subplots(1, 2 if self.model_type == 'vae' else 3, figsize=(15, 5))
        
        if self.model_type == 'vae':
            # VAE losses
            axes[0].plot(history['train_loss'], label='Train')
            axes[0].plot(history['val_loss'], label='Validation')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training History')
            axes[0].legend()
            
            axes[1].plot(history['recon_loss'], label='Reconstruction')
            axes[1].plot(history['kl_loss'], label='KL Divergence')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Loss Components')
            axes[1].legend()
        
        else:  # GAN
            # Generator and Discriminator losses
            axes[0].plot(history['g_loss'], label='Generator')
            axes[0].plot(history['d_loss'], label='Discriminator')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('GAN Training Losses')
            axes[0].legend()
            
            # Discriminator scores
            axes[1].plot(history['d_real'], label='D(real)')
            axes[1].plot(history['d_fake'], label='D(fake)')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Score')
            axes[1].set_title('Discriminator Scores')
            axes[1].legend()
            axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            
            # Loss difference
            axes[2].plot(np.array(history['g_loss']) - np.array(history['d_loss']))
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('G Loss - D Loss')
            axes[2].set_title('Generator-Discriminator Balance')
            axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_type}_training_history.png', dpi=150)
        plt.show()


def demonstrate_inverse_design():
    """Demonstrate inverse design workflow"""
    
    print("=== Inverse Design Demonstration ===\n")
    
    # Train VAE model
    print("Training Conditional VAE model...")
    trainer = InverseDesignTrainer(model_type='vae')
    trainer.load_data('ml_ready_data')
    history = trainer.train_vae(epochs=50, batch_size=64)
    trainer.visualize_training(history)
    
    # Example inverse design problem
    print("\n=== Inverse Design Example ===")
    print("Desired Performance:")
    desired_outputs = {
        'tensile_strength': 3500,  # N
        'contact_resistance': 12,  # μΩ
        'cycles_to_failure': 1000,  # cycles
        'quality_score': 0.85
    }
    
    for key, value in desired_outputs.items():
        print(f"  {key}: {value}")
    
    # Generate solutions
    print("\nGenerating welding parameter solutions...")
    solutions = trainer.inverse_design(desired_outputs, n_solutions=20)
    
    print("\nTop 5 Solutions:")
    print(solutions.head())
    
    # Save model
    trainer.save_model('models')
    
    # Train GAN model for comparison
    print("\n\nTraining Conditional GAN model...")
    gan_trainer = InverseDesignTrainer(model_type='gan')
    gan_trainer.load_data('ml_ready_data')
    gan_history = gan_trainer.train_gan(epochs=100, batch_size=64)
    gan_trainer.visualize_training(gan_history)
    
    # Generate solutions with GAN
    print("\nGAN-based Solutions:")
    gan_solutions = gan_trainer.inverse_design(desired_outputs, n_solutions=20)
    print(gan_solutions.head())
    
    gan_trainer.save_model('models_gan')
    
    return trainer, gan_trainer, solutions, gan_solutions


if __name__ == "__main__":
    # Ensure data is generated first
    if not os.path.exists('ml_ready_data'):
        print("Please run dataset generation and analysis first!")
    else:
        trainer, gan_trainer, vae_solutions, gan_solutions = demonstrate_inverse_design()