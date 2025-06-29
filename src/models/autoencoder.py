import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import cv2
import os

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_shape=(128, 128, 1)):
        super().__init__()
        self.input_shape = input_shape
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.autoencoder = nn.Sequential(self.encoder, self.decoder)
        
    def _build_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0)
        )
    
    def _build_decoder(self):
        return nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.autoencoder(x)
    
    def train_model(self, train_data, epochs=50, batch_size=32, validation_split=0.2):
        # Convert to tensor and create dataloader
        tensor_data = torch.tensor(train_data, dtype=torch.float32).permute(0, 3, 1, 2)
        dataset = TensorDataset(tensor_data, tensor_data)
        
        # Train-validation split
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        
        optimizer = optim.Adam(self.parameters())
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                x, y = batch
                optimizer.zero_grad()
                outputs = self(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    outputs = self(x)
                    val_loss += criterion(outputs, y).item()
            
            print(f'Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}')
    
    def detect_anomalies(self, test_data, threshold=0.03):
        self.eval()
        tensor_data = torch.tensor(test_data, dtype=torch.float32).permute(0, 3, 1, 2)
        
        with torch.no_grad():
            reconstructions = self(tensor_data).detach().numpy()
        
        # Convert back to channel-last format
        reconstructions = reconstructions.transpose(0, 2, 3, 1)
        mse = np.mean(np.square(test_data - reconstructions), axis=(1, 2, 3))
        return mse > threshold, mse
