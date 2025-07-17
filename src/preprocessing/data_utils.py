"""
Utility functions for data loading and preprocessing.
"""
import os
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image


def process_single_image(filename: str) -> np.ndarray:
    """Loads a single image, converts it to grayscale, and flattens it.

    Args:
        filename (str): The path to the image file.

    Returns:
        np.ndarray: A 1D numpy array representing the flattened image.
    """
    print(f"Loading image {filename}")
    img = Image.open(filename).convert('L')  # Convert to grayscale
    img = img.resize((128, 128))  # Resize for consistency
    img_array = np.array(img).flatten()  # Flatten the image into a vector
    return img_array


def load_images_as_vectors(image_folder: str) -> tuple[np.ndarray, list[str]]:
    """Loads all images from a folder in parallel and converts them to vectors.

    Args:
        image_folder (str): The path to the folder containing images.

    Returns:
        tuple[np.ndarray, list[str]]: A tuple containing a 2D numpy array of
        image vectors and a list of corresponding image filenames.
    """
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".png", ".jpg"))]
    image_names = [os.path.basename(path) for path in image_paths]
    image_paths.sort()
    image_names.sort()

    # Use multiprocessing to load images in parallel
    with Pool(cpu_count()) as pool:
        image_vectors = pool.map(process_single_image, image_paths)

    return np.array(image_vectors), image_names
