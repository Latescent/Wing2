"""
This module provides the NoiseDetector class for identifying and visualizing noise in images.

It offers both traditional computer vision methods and deep learning-based approaches
for noise detection. The module is designed to process images in batches and
can be used to replicate experiments described in an accompanying manuscript.
"""

import os
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


class NoiseDetector:
    """Detects noise in images using various methods."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initializes the NoiseDetector.

        Args:
            device (str): The device to use for PyTorch models ('cuda' or 'cpu').
        """
        self.device = device
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )

    def calculate_contour_noise(self, img_path: str) -> int:
        """Calculates noise based on the number of small contours in an image.

        This method implements a traditional computer vision approach where noise
        is quantified by counting the number of contours with an area smaller
        than a predefined threshold. This corresponds to Eq.1 in the manuscript.

        Args:
            img_path (str): The path to the input image file.

        Returns:
            int: The number of small contours, representing the noise level.
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return sum(1 for cnt in contours if cv2.contourArea(cnt) < 50)

    def calculate_deep_noise(self, img_path: str, model: torch.nn.Module) -> float:
        """Calculates noise using a deep learning model's reconstruction error.

        This method uses a PyTorch-based model (e.g., an autoencoder) to
        reconstruct an image. The mean squared error between the original and
        the reconstructed image is used as the noise score.

        Args:
            img_path (str): The path to the input image file.
            model (torch.nn.Module): The trained PyTorch model for noise detection.

        Returns:
            float: The reconstruction loss, representing the noise level.
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            reconstruction = model(img_tensor)
            loss = F.mse_loss(reconstruction, img_tensor).item()

        return loss

    def batch_process(
        self,
        img_dir: str,
        model: torch.nn.Module = None,
        threshold: int = 3500,
        use_deep: bool = False,
    ) -> list[str]:
        """Processes a directory of images to find noisy ones.

        This method uses a hybrid approach to detect noise, allowing for either
        the contour-based or deep learning-based method. It processes images
        in parallel and provides progress tracking.

        Args:
            img_dir (str): The directory containing the images to process.
            model (torch.nn.Module, optional): The deep learning model to use if
                `use_deep` is True. Defaults to None.
            threshold (int, optional): The score threshold above which an image
                is considered noisy. Defaults to 3500.
            use_deep (bool, optional): If True, uses the deep learning method;
                otherwise, uses the contour method. Defaults to False.

        Returns:
            list[str]: A list of paths to the images identified as noisy.
        """
        img_paths = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith((".png", ".jpg"))
        ]

        noisy_paths = []

        with Pool() as pool:
            if use_deep:
                results = list(
                    tqdm(
                        pool.imap(
                            lambda x: (x, self.calculate_deep_noise(x, model)),
                            img_paths,
                        ),
                        total=len(img_paths),
                        desc="Deep Noise Analysis",
                    )
                )
            else:
                results = list(
                    tqdm(
                        pool.imap(
                            lambda x: (x, self.calculate_contour_noise(x)), img_paths
                        ),
                        total=len(img_paths),
                        desc="Contour Noise Analysis",
                    )
                )

            for path, score in results:
                if score > threshold:
                    noisy_paths.append(path)

        return noisy_paths

    def visualize_noise(self, img_path: str, save_path: str = None) -> np.ndarray:
        """Highlights small contours in an image to visualize noise.

        This method creates a visual representation of the noise by drawing
        red boxes around small contours, corresponding to Figure 3 in the manuscript.

        Args:
            img_path (str): The path to the input image file.
            save_path (str, optional): The path to save the output image.
                If None, the image is not saved. Defaults to None.

        Returns:
            np.ndarray: The output image with noise visualized as an array.
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Find and draw small contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                cv2.drawContours(output, [cnt], -1, (0, 0, 255), 1)

        if save_path:
            cv2.imwrite(save_path, output)

        return output


if __name__ == "__main__":
    # Example usage matching manuscript experiments
    detector = NoiseDetector()

    # Traditional method (Section 2.2)
    noisy_images = detector.batch_process(
        "../../data/raw", threshold=3500, use_deep=False
    )

    # Visualization for manuscript figures
    detector.visualize_noise(
        "../../data/raw/AT-0001-031-003679-R.dw.png",
        "../../reports/noise_visualization.png",
    )
