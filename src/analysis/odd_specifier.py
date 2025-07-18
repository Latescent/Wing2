"""
Classifies images as clean or noisy using two distinct methods.

An Isolation Forest-based method that extracts feature scores (based on
contours and edges) from images and uses the model to classify them as
inliers (clean) or outliers (noisy).
"""

import concurrent.futures
import os
import shutil
import sys
import yaml

import cv2
import numpy as np
from scipy.spatial import distance
from skimage import filters
from sklearn.ensemble import IsolationForest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.data_utils import get_image_paths


counter = 0


def progress_bar(len: int, counter: int, txt: str = "Loading:"):
    """
    Displays a simple text-based progress bar in the console.

    Args:
        length (int): The total number of items for 100%.
        count (int): The current number of completed items.
        txt (str, optional): The text to display before the bar.
            Defaults to "Loading:".
    """
    percentage = int(counter * 100 / len)
    print(
        f"{txt} |{'=' * percentage}{'-' * (100 - percentage)}| {percentage}%", end="\r"
    )
    if counter == len:
        sys.stdout.write(f"\r{txt} 100%\033[K\n")


def get_contour_noise_score(image: np.ndarray, area_threshold: int) -> int:
    """Calculates a noise score based on the number of small contours.

    This method is useful for detecting noise such as small, disconnected
    artifacts or breaks in larger structures (e.g., "broken wings").

    Args:
        image (np.ndarray): A grayscale input image as a NumPy array.
        area_threshold (int): The area in pixels below which a contour is
            considered noise.

    Returns:
        int: The number of contours with an area less than 50 pixels.
    """
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    noise_score = sum(1 for contour in contours if cv2.contourArea(contour) < area_threshold)
    return noise_score


def get_edge_noise_score(image: np.ndarray) -> float:
    """Calculates a noise score based on the standard deviation of edge intensity.

    This method uses a Sobel filter to detect edges. A higher standard deviation
    can indicate more complex or jagged edges, which may be characteristic of
    noise or unusual features like forks in wings.

    Args:
        image (np.ndarray): A grayscale input image as a NumPy array.

    Returns:
        float: The standard deviation of the Sobel edge-detected image.
    """
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    edges = filters.sobel(binary_image)
    return float(np.std(edges))


def extract_features(image_path: str, contour_threshold: int) -> tuple[int, float]:
    """Extracts a feature tuple from a single image using both noise methods.

    Args:
        image_path (str): The file path to the image.
        contour_threshold (int): The area threshold to use for contour noise detection.

    Returns:
        tuple[int, float]: A tuple containing the contour noise score and the
        edge noise score.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noise_score_contours = get_contour_noise_score(image, contour_threshold)
    noise_score_edge = get_edge_noise_score(image)
    return (noise_score_contours, noise_score_edge)


def run_anomaly_detection(image_paths: list[str], contamination: float,
                          contour_threshold: int) -> tuple[list[str], np.ndarray]:
    """Trains an Isolation Forest model and predicts anomalies.

    This function orchestrates the loading of images, parallel feature
    extraction, and training of the model to predict which images are outliers.

    Args:
        image_paths (list[str]): A list of paths to the images to process.
        contamination (float): The expected proportion of outliers in the data.
        contour_threshold (int): The area threshold for contour noise detection.

    Returns:
        tuple[list[str], np.ndarray]: A tuple containing the list of image
        paths and the corresponding model predictions (-1 for outliers, 1 for
        inliers).
    """
    print("Extracting features in parallel...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        features = list(executor.map(lambda p: extract_features(p, contour_threshold), image_paths))
    print("Feature extraction complete.")

    print("Training Isolation Forest model...")
    features_array = np.array(features)
    iso_forest = IsolationForest(
        contamination=contamination, #type: ignore
        random_state=42,
        n_jobs=-1
    )
    predictions = iso_forest.fit_predict(features_array)  # fit the Isolation Forest
    print("Model training complete.")

    return image_paths, predictions


def classify_images_by_anomaly(output_base_dir: str, image_paths: list[str], predictions: np.ndarray):
    """Classifies images into 'clean' and 'noisy' folders using the model.

    Args:
        output_base_dir (str): The root directory to save the output folders.
        image_paths (list[str]): The list of original image paths.
        predictions (np.ndarray): The prediction labels from the Isolation Forest model.
    """
    clean_dir = os.path.join(output_base_dir, "iso_forest_clean")
    noisy_dir = os.path.join(output_base_dir, "iso_forest_noisy")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    # Classify images based on model's output
    for i, image_path in enumerate(image_paths):
        if predictions[i] == 1:  # Inliers are considered 'clean'
            dest_path = os.path.join(clean_dir, os.path.basename(image_path))
        else:  # Outliers (-1) are considered 'noisy'
            dest_path = os.path.join(noisy_dir, os.path.basename(image_path))

        shutil.copy(image_path, dest_path)

    print(
        "\nClassification complete. Images have been sorted into clean and noisy folders."
    )


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Load Configuration ---
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/', 'config.yaml')
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get general paths and method-specific parameters
    image_folder = config["data_dir"]
    output_folder = config["output_dir"]
    params = config["classification_analysis"]
    contamination = params["contamination"]
    contour_threshold = params["contour_area_threshold"]

    # --- Execute Isolation Forest Workflow ---
    print("Starting Isolation Forest classification...")
    all_image_paths = get_image_paths(image_folder)

    # Pass parameters down to the function
    image_paths_processed, predictions = run_anomaly_detection(
        all_image_paths, 
        contamination, 
        contour_threshold
    )

    # Pass output folder and results to the saving function
    classify_images_by_anomaly(
        output_folder, 
        image_paths_processed, 
        predictions
    )
