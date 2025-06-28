import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
import cv2
from tqdm import tqdm

class ConvolutionalAutoencoder:
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.autoencoder, self.encoder, self.decoder = self._build_model()
        
    def _build_model(self):
        # Encoder
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder
        x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(encoded)
        x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        # Models
        autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)
        
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder, encoder, decoder

    def train(self, train_data, epochs=50, batch_size=32):
        self.autoencoder.fit(
            train_data, train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
    
    def detect_anomalies(self, test_data, threshold=0.03):
        reconstructions = self.autoencoder.predict(test_data)
        mse = np.mean(np.square(test_data - reconstructions), axis=(1,2,3))
        return mse > threshold, mse