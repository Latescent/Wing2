import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WingImageDataset(Dataset):
    """
    Dataset loader for wing image data preprocessing and normalization.
    """
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load grayscale image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Normalize pixel intensities to [0, 1] range
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension for PyTorch compatibility
        image = np.expand_dims(image, axis=0)
        
        if self.transform:
            image = self.transform(image)
            
        return torch.from_numpy(image), image_path

class ConvolutionalAutoencoder(nn.Module):
    """
    Convolutional autoencoder architecture for wing image reconstruction.
    
    Encoder: Five convolutional layers with ascending filter dimensions
    Decoder: Five transposed convolutional layers with descending filter dimensions
    Latent space: 128-dimensional compressed representation
    """
    
    def __init__(self, input_channels=1, latent_dim=128):
        super(ConvolutionalAutoencoder, self).__init__()
        
        # Encoder layers with batch normalization and ReLU activation
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Linear projection to latent space
            nn.Linear(512, latent_dim)
        )
        
        # Decoder fully connected layer
        self.decoder_fc = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Decoder layers with transposed convolutions
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),
            
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Output layer with sigmoid activation for [0,1] normalization
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim
        
    def forward(self, x):
        # Encode input to latent representation
        encoded = self.encoder(x)
        
        # Decode to reconstruct input
        decoded_fc = self.decoder_fc(encoded)
        reconstructed = self.decoder(decoded_fc)
        
        return reconstructed, encoded

class AnomalyDetector:
    """
    Autoencoder-based anomaly detection system for wing image quality assessment.
    
    Implements reconstruction error analysis with dynamic threshold determination
    based on statistical deviation from normal reconstruction patterns.
    """
    
    def __init__(self, model_save_path='autoencoder_model.pth', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ConvolutionalAutoencoder().to(self.device)
        self.model_save_path = model_save_path
        self.reconstruction_errors = []
        self.threshold = None
        
        logger.info(f"Initialized autoencoder with {self._count_parameters()} parameters")
        logger.info(f"Using device: {self.device}")
    
    def _count_parameters(self):
        """Count trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train(self, train_loader, val_loader=None, epochs=None, learning_rate=None, 
              patience_scheduler=None, patience_early_stopping=None, delta=None):
        """
        Train autoencoder with Adam optimizer and learning rate scheduling.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Maximum number of training epochs
            learning_rate: Initial learning rate for Adam optimizer
            patience_scheduler: Patience for learning rate reduction
            patience_early_stopping: Patience for early stopping
            delta: Minimum change threshold for early stopping
        """
        # Initialize Adam optimizer with specified hyperparameters
        optimizer = optim.Adam(self.model.parameters(), 
                             lr=learning_rate, 
                             betas=(0.9, 0.999))
        
        # Learning rate scheduler with plateau detection
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min',
                                                        patience=patience_scheduler,
                                                        factor=0.5,
                                                        verbose=True)
        
        # Mean squared error loss for pixel-wise reconstruction
        criterion = nn.MSELoss()
        
        # Early stopping implementation
        best_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        logger.info("Starting training...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
                data = data.to(self.device)
                
                optimizer.zero_grad()
                reconstructed, _ = self.model(data)
                loss = criterion(reconstructed, data)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for data, _ in val_loader:
                        data = data.to(self.device)
                        reconstructed, _ = self.model(data)
                        loss = criterion(reconstructed, data)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_loss - delta:
                    bestLoss = avg_val_loss
                    patience_counter = 0
                    torch.save({'epoch': epoch,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': bestLoss,},self.model_save_path)
                else:
                    patience_counter += 1
                
                logger.info(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
                
                if patience_counter >= patience_early_stopping:
                    logger.info(f'Early stopping at epoch {epoch+1}')
                    break
            else:
                logger.info(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}')
        
        if val_loader and os.path.exists(self.model_save_path):
            checkpoint = torch.load(self.model_save_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded best model weights")
        
        return train_losses, val_losses
    # ... rest of AnomalyDetector unchanged ...

def main():
    """
    Main execution pipeline for autoencoder-based anomaly detection.
    """
    # Load configuration from YAML
    with open('configs/autoencoder_params.yaml') as f:
        config = yaml.safe_load(f)

    # Collect image paths from directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(config['data_dir']).glob(f'**/*{ext}'))
    image_paths = [str(path) for path in image_paths]

    # Create train/validation split
    np.random.seed(42)
    np.random.shuffle(image_paths)
    split_idx = int(len(image_paths) * config['train_split'])
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]

    train_dataset = WingImageDataset(train_paths)
    val_dataset = WingImageDataset(val_paths)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'], shuffle=False, num_workers=4)

    detector = AnomalyDetector(model_save_path=config['model_save_path'])
    train_losses, val_losses = detector.train(
        train_loader, val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        patience_scheduler=config['reduce_lr_patience'],
        patience_early_stopping=config['early_stopping_patience'],
        delta=config['early_stopping_delta']
    )

    detector.compute_reconstruction_errors(train_loader)
    detector.set_threshold(sigma_multiplier=3)
    detector.plot_reconstruction_errors(save_path='reconstruction_errors.png')

    if os.path.exists(config['expert_annotations']):
        detector.evaluate_on_expert_annotations(config['expert_annotations'], val_loader)

    all_loader = DataLoader(WingImageDataset(image_paths), batch_size=config['batch_size'], shuffle=False)
    normal_images, anomalous_images, stats = detector.filter_dataset(image_paths, config['output_dir'])

    logger.info("Anomaly detection pipeline completed successfully")
    logger.info(f"Results saved to: {config['output_dir']}")

if __name__ == "__main__":
    main()
