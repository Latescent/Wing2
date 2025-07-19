import json
import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WingImageDataset(Dataset):
    """Creates a dataset for loading, preprocessing, and normalizing wing images.

    This class handles loading grayscale images, normalizing pixel intensities
    to a [0, 1] range, and ensuring they have the correct dimensions for
    PyTorch models.
    """

    def __init__(self, image_paths: list[str], transform=None):
        """Initializes the dataset.

        Args:
            image_paths (list[str]): A list of file paths to the images.
            transform (callable, optional): An optional transform to be applied
                on a sample. Defaults to None.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """Gets a single image and its path from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple[torch.Tensor, str]: A tuple containing the processed image
            tensor and its original file path.

        Raises:
            ValueError: If an image cannot be loaded from the specified path.
        """
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
    """Defines a convolutional autoencoder architecture.

    The model is designed for wing image reconstruction.

    Architecture:
        Encoder: Compresses the input image using five convolutional layers.
        Decoder: Reconstructs the image from the latent space using five
            transposed convolutional layers.
        Latent Space: A 128-dimensional compressed vector representation.
    """
    def __init__(self, input_channels: int = 1, latent_dim: int = 128):
        """Initializes the autoencoder model layers.

        Args:
            input_channels (int, optional): The number of channels in the input
                image. Defaults to 1 for grayscale.
            latent_dim (int, optional): The dimensionality of the latent space.
                Defaults to 128.
        """
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
            nn.Linear(512, latent_dim),
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
            nn.Sigmoid(),
        )

        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the autoencoder.

        Args:
            x (torch.Tensor): The input batch of images.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the
            reconstructed images and their corresponding latent space vectors.
        """
        # Encode input to latent representation
        encoded = self.encoder(x)

        # Decode to reconstruct input
        decoded_fc = self.decoder_fc(encoded)
        reconstructed = self.decoder(decoded_fc)

        return reconstructed, encoded


class AnomalyDetector:
    """Manages the autoencoder-based anomaly detection system.

    This class handles model training, evaluation, and the use of reconstruction
    error for anomaly detection in wing images. It implements dynamic threshold
    determination based on statistical deviation from normal reconstruction patterns.
    """

    def __init__(self, model_save_path: str = "autoencoder_model.pth", device=None):
        """Initializes the AnomalyDetector.

        Args:
            model_save_path (str, optional): Path to save or load the trained
                model. Defaults to 'autoencoder_model.pth'.
            device (torch.device, optional): The device to run the model on.
                If None, automatically selects CUDA if available. Defaults to None.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvolutionalAutoencoder().to(self.device)
        self.model_save_path = model_save_path
        self.reconstruction_errors = []
        self.threshold = 0.0

        logger.info(f"Initialized autoencoder with {self._count_parameters()} parameters")
        logger.info(f"Using device: {self.device}")

    def _count_parameters(self) -> int:
        """Counts the number of trainable parameters in the model.

        Returns:
            int: The total count of trainable parameters.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 0,
              learning_rate: float = 0, patience_scheduler: int = 0,
              patience_early_stopping: int = 0, delta: float = 0):
        """Trains the autoencoder with early stopping and learning rate scheduling.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
                Defaults to None.
            epochs (int, optional): The maximum number of training epochs.
            learning_rate (float, optional): The initial learning rate for the Adam
                optimizer.
            patience_scheduler (int, optional): Number of epochs with no improvement
                after which learning rate will be reduced.
            patience_early_stopping (int, optional): Number of epochs with no
                improvement after which training will be stopped.
            delta (float, optional): Minimum change in the monitored quantity to
                qualify as an improvement for early stopping.

        Returns:
            tuple[list[float], list[float]]: A tuple containing the lists of
            training and validation losses for each epoch.
        """
        # Initialize Adam optimizer with specified hyperparameters
        optimizer = optim.Adam(self.model.parameters(),
                               lr=learning_rate,
                               betas=(0.9, 0.999))

        # Learning rate scheduler with plateau detection
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=patience_scheduler, factor=0.5)

        # Mean squared error loss for pixel-wise reconstruction
        criterion = nn.MSELoss()

        # Early stopping implementation
        best_loss = float("inf")
        patience_counter = 0

        train_losses = []
        val_losses = []

        logger.info("Starting training...")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for _, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
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
                    best_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": best_loss,
                        },
                        self.model_save_path,
                    )
                else:
                    patience_counter += 1

                logger.info(
                    f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                )

                if patience_counter >= patience_early_stopping:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                logger.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}")

        if val_loader and os.path.exists(self.model_save_path):
            checkpoint = torch.load(self.model_save_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded best model weights")

        return train_losses, val_losses

    #NOTE: Start of the missing implementations

    def compute_reconstruction_errors(self, data_loader: DataLoader):
        """Computes the reconstruction error for each image in a dataset.

        Args:
            data_loader (DataLoader): DataLoader for the dataset to be evaluated.
        """
        logger.info("Computing reconstruction errors...")
        self.model.eval()
        errors = []
        criterion = nn.MSELoss(reduction='none')

        with torch.no_grad():
            for data, _ in tqdm(data_loader, desc="Calculating Errors"):
                data = data.to(self.device)
                reconstructed, _ = self.model(data)
                # Calculate loss for each item in the batch
                loss = criterion(reconstructed, data)
                # Reduce loss to a single value per image (e.g., mean over pixels)
                per_image_loss = torch.mean(loss.view(loss.size(0), -1), dim=1)
                errors.extend(per_image_loss.cpu().numpy())
        
        self.reconstruction_errors = errors
        logger.info(f"Computed errors for {len(errors)} images.")


    def set_threshold(self, sigma_multiplier: float = 3.0):
        """Sets the anomaly detection threshold based on reconstruction errors.

        The threshold is defined as mean + (sigma_multiplier * std_dev).

        Args:
            sigma_multiplier (float, optional): The number of standard deviations
                to use for the threshold. Defaults to 3.0.
        """
        if not self.reconstruction_errors:
            raise ValueError("Reconstruction errors have not been computed. Call compute_reconstruction_errors first.")

        mean_error = np.mean(self.reconstruction_errors)
        std_error = np.std(self.reconstruction_errors)
        self.threshold = mean_error + (sigma_multiplier * std_error)

        logger.info(f"Anomaly threshold set to: {self.threshold:.6f}")


    def plot_reconstruction_errors(self, save_path: str = 'reconstruction_errors.png'):
        """Plots a histogram of reconstruction errors and the anomaly threshold.

        Args:
            save_path (str, optional): The path to save the plot image.
                Defaults to 'reconstruction_errors.png'.
        """
        if not self.reconstruction_errors or self.threshold is None:
            raise ValueError("Errors and threshold must be computed first.")

        plt.figure(figsize=(10, 6))
        plt.hist(self.reconstruction_errors, bins=50, alpha=0.75, label='Error Distribution')
        plt.axvline(float(self.threshold), color='r', linestyle='--', label=f'Threshold ({self.threshold:.4f})')
        plt.title('Distribution of Reconstruction Errors')
        plt.xlabel('Reconstruction Error (MSE)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Reconstruction error plot saved to {save_path}")
        plt.show()


    def evaluate_on_expert_annotations(self, annotations_path: str, data_loader: DataLoader):
        """Evaluates the model's performance against expert annotations.

        Args:
            annotations_path (str): Path to a JSON file with expert annotations.
                Expected format: {"image_name.png": "normal" or "anomalous"}.
            data_loader (DataLoader): DataLoader for the dataset to be evaluated.
        """
        logger.info("Evaluating performance against expert annotations...")
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        self.model.eval()
        y_true, y_pred = [], []
        criterion = nn.MSELoss()

        with torch.no_grad():
            for data, paths in tqdm(data_loader, desc="Evaluating"):
                data = data.to(self.device)
                reconstructed, _ = self.model(data)
                
                for i, path in enumerate(paths):
                    image_name = os.path.basename(path)
                    if image_name in annotations:
                        # Convert string labels to binary (1 for normal, 0 for anomalous)
                        true_label = 1 if annotations[image_name] == 'normal' else 0
                        
                        # Calculate reconstruction error for the single image
                        error = criterion(reconstructed[i], data[i]).item()
                        pred_label = 1 if error < self.threshold else 0
                        
                        y_true.append(true_label)
                        y_pred.append(pred_label)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        logger.info(f"Evaluation Metrics:\n"
                    f"  Accuracy:  {accuracy:.4f}\n"
                    f"  Precision: {precision:.4f}\n"
                    f"  Recall:    {recall:.4f}\n"
                    f"  F1-Score:  {f1:.4f}")


    def filter_dataset(self, image_paths: list[str], output_dir: str) -> tuple[list, list, dict]:
        """Filters a dataset into normal and anomalous images based on the threshold.

        Args:
            image_paths (list[str]): A list of all image paths to filter.
            output_dir (str): The root directory to save the sorted images.

        Returns:
            tuple[list, list, dict]: A tuple containing a list of normal image paths,
            a list of anomalous image paths, and a dictionary of statistics.
        """
        logger.info("Filtering dataset into normal and anomalous folders...")
        normal_dir = os.path.join(output_dir, 'normal')
        anomalous_dir = os.path.join(output_dir, 'anomalous')
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(anomalous_dir, exist_ok=True)
        
        dataset = WingImageDataset(image_paths)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        self.model.eval()
        normal_images, anomalous_images = [], []
        
        with torch.no_grad():
            for data, paths in tqdm(data_loader, desc="Filtering Images"):
                data = data.to(self.device)
                reconstructed, _ = self.model(data)
                
                for i, path in enumerate(paths):
                    error = torch.mean((reconstructed[i] - data[i])**2).item()
                    
                    if error < self.threshold:
                        normal_images.append(path)
                        shutil.copy(path, os.path.join(normal_dir, os.path.basename(path)))
                    else:
                        anomalous_images.append(path)
                        shutil.copy(path, os.path.join(anomalous_dir, os.path.basename(path)))
        
        stats = {
            "total_images": len(image_paths),
            "normal_count": len(normal_images),
            "anomalous_count": len(anomalous_images)
        }
        logger.info(f"Filtering complete: {stats}")
        return normal_images, anomalous_images, stats

    #NOTE: End of the missing implementations


def main():
    """
    Main execution pipeline for autoencoder-based anomaly detection.
    """
    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/', 'config.yaml')
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    params = config["autoencoder"]

    # Collect image paths from directory
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(config["data_dir"]).glob(f"**/*{ext}"))
    image_paths = [str(path) for path in image_paths]

    # Create train/validation split
    np.random.seed(42)
    np.random.shuffle(image_paths)
    split_idx = int(len(image_paths) * params["train_split"])
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    #WARN: parameters valid
    train_dataset = WingImageDataset(train_paths)
    val_dataset = WingImageDataset(val_paths)
    #WARN: Fed to torch util
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=-1
    )
    val_loader = DataLoader(
        val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=-1
    )
    #WARN: save path added
    detector = AnomalyDetector(model_save_path=params["model_save_path"])
    #WARN: checking...
    train_losses, val_losses = detector.train(
        train_loader,
        val_loader,
        epochs=params["epochs"],
        learning_rate=params["learning_rate"],
        patience_scheduler=params["reduce_lr_patience"],
        patience_early_stopping=params["early_stopping_patience"],
        delta=params["early_stopping_delta"],
    )

    #NOTE: The following methods were not defined in the previous version of the script.
    #NOTE: The current definition is based on the use case provided below
    #NOTE: Check to see if the implementations follow the methods provided in the manuscript

    detector.compute_reconstruction_errors(train_loader)
    detector.set_threshold(sigma_multiplier=3)
    detector.plot_reconstruction_errors(save_path="reconstruction_errors.png")

    if os.path.exists(params["expert_annotations"]):
        detector.evaluate_on_expert_annotations(
            params["expert_annotations"], val_loader
        )

    all_loader = DataLoader(
        WingImageDataset(image_paths), batch_size=params["batch_size"], shuffle=False
    )
    normal_images, anomalous_images, stats = detector.filter_dataset(
        image_paths, config["output_dir"]
    )

    logger.info("Anomaly detection pipeline completed successfully")
    logger.info(f"Results saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()

