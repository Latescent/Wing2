"""
Classifies images as clean or noisy using two distinct methods.

This script contains two primary workflows:
1.  An Isolation Forest-based method that extracts feature scores (based on
    contours and edges) from images and uses the model to classify them as
    inliers (clean) or outliers (noisy).
2.  An intersection-based method that compares the geometric pattern of
    intersection points across a set of images to identify outliers.

The main execution block currently runs the Isolation Forest classification.
"""

import concurrent.futures
import os
import sys

import cv2
import numpy as np
from scipy.spatial import distance
from skimage import filters
from sklearn.ensemble import IsolationForest

# Allows importing from a sibling directory.
sys.path.insert(0, "../preprocessing/")
from intersections import find_intersections_via_hit_or_miss

# --- Global Configuration ---
input_dir = "../../data/processed_sample/"
clean_dir = "../../data/processed_sample/clean"
noisy_dir = "../../data/processed_sample/noisy"

os.makedirs(clean_dir, exist_ok=True)
os.makedirs(noisy_dir, exist_ok=True)

counter = 0


def load_images() -> list[str]:
    """Loads all image paths from the global input directory.

    Returns:
        list[str]: A sorted list of full paths to the images.
    """
    global input_dir, clean_dir, noisy_dir
    images = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    images.sort()
    return images


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


# ==============================================================================
# --- Part 1: Isolation Forest Based Classification ---
# ==============================================================================


def get_contour_noise_score(image: np.ndarray) -> int:
    """Calculates a noise score based on the number of small contours.

    This method is useful for detecting noise such as small, disconnected
    artifacts or breaks in larger structures (e.g., "broken wings").

    Args:
        image (np.ndarray): A grayscale input image as a NumPy array.

    Returns:
        int: The number of contours with an area less than 50 pixels.
    """
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    noise_score = sum(1 for contour in contours if cv2.contourArea(contour) < 50)
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
    std_edge = np.std(edges)
    return std_edge


def extract_features(image_path: str) -> tuple[int, float]:
    """Extracts a feature tuple from a single image using both noise methods.

    Args:
        image_path (str): The file path to the image.

    Returns:
        tuple[int, float]: A tuple containing the contour noise score and the
        edge noise score.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noise_score_contours = get_contour_noise_score(image)
    noise_score_edge = get_edge_noise_score(image)
    return (noise_score_contours, noise_score_edge)


def _process_image_features(image_name: str) -> tuple[tuple[int, float], str]:
    """A wrapper for parallel feature extraction.

    Args:
        image_path (str): The file path to the image.

    Returns:
        tuple[tuple[int, float], str]: A tuple containing the feature vector
        and the corresponding image path.
    """
    image_path = os.path.join(input_dir, image_name)
    image_features = extract_features(image_path)
    print(f"Extracted features of {image_path}")
    return image_features, image_path


def run_anomaly_detection():
    """Trains an Isolation Forest model to classify images.

    This function orchestrates the loading of images, parallel feature
    extraction, and training of the model to predict which images are outliers.

    Returns:
        tuple[list[str], np.ndarray]: A tuple containing the list of image
        paths and the corresponding model predictions (-1 for outliers, 1 for
        inliers).
    """
    images = load_images()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(_process_image_features, images))

    print("\nPreparing model")
    features = [result[0] for result in results]
    image_paths = [result[1] for result in results]
    print("\nModel prepared")
    features_array = np.array(features)
    print("\nTraining")

    # The IsolationForest model identifies anomalies in the feature space.
    iso_forest = IsolationForest(contamination=0.2, random_state=42, n_jobs=-1)
    predictions = iso_forest.fit_predict(features_array)  # fit the Isolation Forest
    print("\nModel trained. Saving the images")

    return image_paths, predictions


def classify_images_by_anomaly():
    """Classifies images into 'clean' and 'noisy' folders using the model."""
    image_paths, predictions = run_anomaly_detection()

    # Classify images based on model's output
    for i, image_path in enumerate(image_paths):
        if predictions[i] == 1:  # Inliers are considered 'clean'
            dest_path = os.path.join(clean_dir, os.path.basename(image_path))
        else:  # Outliers (-1) are considered 'noisy'
            dest_path = os.path.join(noisy_dir, os.path.basename(image_path))

        cv2.imwrite(dest_path, cv2.imread(image_path))

    print(
        "\nClassification complete. Images have been sorted into clean and noisy folders."
    )


# ==============================================================================
# --- Part 2: Intersection-Based Outlier Detection ---
# ==============================================================================


def extract_all_intersections() -> list[tuple[str, list]]:
    """Generates a dataset of intersection coordinates for each image.

    Uses a thread pool to process images in parallel, finding all intersection
    points for each one.

    Returns:
        list[tuple[str, list]]: A list of tuples, where each tuple contains
        the image basename and a list of its intersection (x, y) coordinates.
    """
    images = load_images()

    print("Extracting image intersections:")

    def single_image_intersections(image_path: str) -> tuple[str, list]:
        """Helper function to find intersections for one image."""
        global counter
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        intersections = find_intersections_via_hit_or_miss(image)
        counter += 1
        progress_bar(len(images), counter)
        return (os.path.basename(image_path), intersections)

    def parallel_image_processing() -> list[tuple[str, list]]:
        """Executes intersection extraction in parallel."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = list(executor.map(single_image_intersections, images))
        return result

    return parallel_image_processing()


def min_distance(coord: tuple[int, int], other_object: list[tuple[int, int]]) -> float:
    """Finds the minimum Euclidean distance from a coordinate to a list of coordinates.

    Args:
        coord (tuple[int, int]): The single (x, y) coordinate.
        other_object (list[tuple[int, int]]): A list of other (x, y) coordinates.

    Returns:
        float: The smallest distance from the coordinate to any point in the list.
    """
    return min([distance.euclidean(coord, other_coord) for other_coord in other_object])


def count_matching_points(object1: list, object2: list, threshold: float = 1.0) -> int:
    """Counts how many coordinates in object1 have a close match in object2.

    Args:
        object1 (list): A list of (x, y) coordinates.
        object2 (list): A second list of (x, y) coordinates to compare against.
        threshold (float, optional): The maximum distance to be considered a match.
            Defaults to 1.0.

    Returns:
        int: The total number of coordinates in `object1` that have a match in
        `object2` within the given threshold.
    """
    matched_coords = 0
    for coord in object1:
        if min_distance(coord, object2) < threshold:
            matched_coords += 1
    return matched_coords


def filter_outlier_patterns(
    intersection_patterns: list,
    match_threshold: float = 0.8,
    distance_threshold: int = 10,
) -> list:
    """Identifies and excludes outlier objects based on geometric similarity.

    An object (a list of coordinates from one image) is considered valid if it
    matches with a high percentage of other objects in the list.

    Args:
        intersection_patterns (list): A list where each item is another list of coordinates.
        match_threshold (float, optional): The minimum percentage of coordinates
            that must match for two objects to be considered similar. Defaults to 0.8.
        distance_threshold (int, optional): The distance threshold used when
            comparing individual coordinates. Defaults to 10.

    Returns:
        list: A filtered list containing only the "valid" objects.
    """

    def compare(obj: list) -> list | None:
        """Compares a single object against all others to see if it's an inlier."""
        match_count = 0
        for other_obj in intersection_patterns:
            if obj is not other_obj:
                matched = count_matching_points(obj, other_obj, distance_threshold)
                # If the percentage of matched points exceeds the threshold, count it as a match.
                if matched / len(obj) >= match_threshold:
                    match_count += 1
        # If the object matches with enough other objects, it's considered valid.
        if match_count >= len(intersection_patterns) * match_threshold:
            return obj
        return None

    valid_objects = []
    for obj in intersection_patterns:
        result = compare(obj)
        if result is not None:
            valid_objects.append(result)
    return valid_objects


# --- Main Execution Block ---
if __name__ == "__main__":
    classify_images_by_anomaly()
