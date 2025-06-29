import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import random
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import os

# Load configuration from YAML
with open('configs/autoencoder_params.yaml') as f:
    config = yaml.safe_load(f)

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder, self.decoder = self._build_model()
        self.autoencoder = nn.Sequential(self.encoder, self.decoder)
        
    def _build_encoder(self):
        layers = []
        in_channels = self.config['input_shape'][2]  # Channel last format
        
        # Dynamically build encoder based on config
        for i, filters in enumerate(self.config['encoder_filters']):
            layers.extend([
                nn.Conv2d(in_channels, filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            ])
            in_channels = filters
            
        # Latent space projection
        layers.append(nn.Flatten())
        layers.append(nn.Linear(
            self._calculate_conv_output_size(),
            self.config['latent_dim']
        ))
        
        return nn.Sequential(*layers)
    
    def _calculate_conv_output_size(self):
        # Calculate flattened size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, *self.config['input_shape'])
            dummy = dummy.permute(0, 3, 1, 2)  # to (B, C, H, W)
            for layer in self.encoder[:-2]:  # Exclude flatten and linear
                dummy = layer(dummy)
            return dummy.view(1, -1).shape[1]
    
    def _build_decoder(self):
        # Calculate initial decoder input size
        conv_output_size = self._calculate_conv_output_size()
        spatial_dim = int(np.sqrt(conv_output_size // self.config['encoder_filters'][-1]))
        
        layers = [
            nn.Linear(self.config['latent_dim'], conv_output_size),
            nn.Unflatten(1, (self.config['encoder_filters'][-1], spatial_dim, spatial_dim))
        ]
        
        in_channels = self.config['encoder_filters'][-1]
        
        # Dynamically build decoder based on config
        for i, filters in enumerate(self.config['decoder_filters']):
            layers.extend([
                nn.ConvTranspose2d(in_channels, filters, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU()
            ])
            in_channels = filters
            
        # Final output layer
        layers.extend([
            nn.Conv2d(in_channels, self.config['input_shape'][2], kernel_size=3, padding=1),
            nn.Sigmoid()
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.autoencoder(x)
    
    def train_model(self, train_data):
        # Prepare data
        tensor_data = torch.tensor(train_data, dtype=torch.float32)
        tensor_data = tensor_data.permute(0, 3, 1, 2)  # to (B, C, H, W)
        dataset = TensorDataset(tensor_data, tensor_data)
        
        # Train-validation split
        val_size = int(len(dataset) * self.config['validation_split'])
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_set, 
                                 batch_size=self.config['batch_size'], 
                                 shuffle=True)
        val_loader = DataLoader(val_set, 
                               batch_size=self.config['batch_size'])
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), 
                              lr=0.001, 
                              betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        criterion = nn.MSELoss()
        
        # Training loop
        best_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training phase
            self.train()
            train_loss = 0
            for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}'):
                optimizer.zero_grad()
                outputs = self(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    outputs = self(x)
                    val_loss += criterion(outputs, y).item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_loss - self.config['early_stopping_delta']:
                best_loss = avg_val_loss
                early_stop_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, 'best_model.pth')
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    def detect_anomalies(self, test_data):
        self.eval()
        tensor_data = torch.tensor(test_data, dtype=torch.float32)
        tensor_data = tensor_data.permute(0, 3, 1, 2)  # to (B, C, H, W)
        
        with torch.no_grad():
            reconstructions = self(tensor_data).detach().numpy()
            reconstructions = reconstructions.transpose(0, 2, 3, 1)  # back to (B, H, W, C)
        
        mse = np.mean(np.square(test_data - reconstructions), axis=(1, 2, 3))
        return mse > self.config['anomaly_threshold'], mse

# Extended configuration with training parameters
full_config = {
    'input_shape': [128, 128, 1],
    'encoder_filters': [32, 64],
    'decoder_filters': [64, 32],
    'latent_dim': 32,
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.2,
    'anomaly_threshold': 0.035,
    'learning_rate': 0.001,
    'reduce_lr_factor': 0.5,
    'reduce_lr_patience': 5,
    'early_stopping_patience': 10,
    'early_stopping_delta': 0.001
}

# Save extended config to YAML
with open('configs/autoencoder_params_full.yaml', 'w') as f:
    yaml.dump(full_config, f)

# Example usage
if __name__ == "__main__":
    # Initialize with config
    model = ConvolutionalAutoencoder(full_config)
    
    # Sample data (replace with actual data)
    sample_data = np.random.rand(1000, 128, 128, 1).astype(np.float32)
    
    # Train model
    model.train_model(sample_data)
    
    # Detect anomalies
    test_data = np.random.rand(100, 128, 128, 1).astype(np.float32)
    is_anomaly, mse_scores = model.detect_anomalies(test_data)
