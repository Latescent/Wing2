"""
Performs hierarchical clustering on a directory of images.

This script processes a folder of images by converting them into feature vectors,
reducing their dimensionality with PCA, and then applying hierarchical
clustering to group similar images. The resulting clusters are saved into
separate directories, and a dendrogram can be displayed for visualization.
"""

import os
import shutil
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def process_single_image(filename: str) -> np.ndarray:
    """Loads a single image, converts it to grayscale, and flattens it.

    Args:
        filename (str): The path to the image file.

    Returns:
        np.ndarray: A 1D numpy array representing the flattened image.
    """
    print(f"Loading image {filename}")
    img = Image.open(filename).convert("L")  # Convert to grayscale
    img = img.resize((128, 128))  # Resize for consistency
    img_array = np.array(img).flatten()  # Flatten the image into a vector
    return img_array


def load_images_as_vectors(
    image_folder: str, image_size: tuple[int, int] = (512, 256)
) -> tuple[np.ndarray, list[str]]:
    """Loads all images from a folder in parallel and converts them to vectors.

    Args:
        image_folder (str): The path to the folder containing images.
        image_size (tuple[int, int], optional): The target size for resizing images.
            This argument is included for clarity but is not used in the current
            implementation of process_single_image. Defaults to (512, 256).

    Returns:
        tuple[np.ndarray, list[str]]: A tuple containing a 2D numpy array of
        image vectors and a list of corresponding image filenames.
    """
    image_paths = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.endswith(".png") or f.endswith(".jpg")
    ]
    image_names = [os.path.basename(path) for path in image_paths]
    image_paths.sort()
    image_names.sort()

    # Use multiprocessing to load images in parallel
    with Pool(cpu_count()) as pool:
        image_vectors = pool.map(process_single_image, image_paths)

    return np.array(image_vectors), image_names


def save_images_by_cluster(image_folder: str, image_names: str, clusters: np.ndarray):
    """Saves images into separate folders based on their cluster ID.

    Args:
        image_folder (str): The original directory where the images are stored.
        image_names (list[str]): An ordered list of image filenames.
        clusters (np.ndarray): An array of cluster IDs corresponding to each image.
    """
    output_dir = "/home/delta/Documents/new_database/out_put"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, cluster_id in enumerate(clusters):
        cluster_folder = os.path.join(output_dir, f"cluster_{cluster_id}")
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)

        src_path = os.path.join(image_folder, image_names[i])
        dest_path = os.path.join(cluster_folder, image_names[i])
        shutil.copy(src_path, dest_path)


if __name__ == "__name__":

    # Set the image folder path
    image_folder = "../../data/raw_sample/"

    # Load and preprocess image data
    image_vectors, image_names = load_images_as_vectors(image_folder)

    # Optionally reduce dimensionality using PCA (optional for very large datasets)
    pca = PCA(n_components=50)  # Reduce to 50 components
    image_vectors_pca = pca.fit_transform(image_vectors)

    # Standardize the features
    scaler = StandardScaler()
    image_vectors_scaled = scaler.fit_transform(image_vectors_pca)

    # Perform Hierarchical Clustering using Ward’s method
    linked = linkage(image_vectors_scaled, method="ward")

    # Define the distance threshold or the number of clusters you want to generate
    distance_threshold = 135  # Adjust this value based on your data (e.g., 100)
    clusters = fcluster(linked, t=distance_threshold, criterion="distance")

    # Count the number of unique clusters
    num_clusters = len(np.unique(clusters))
    print(f"Number of clusters identified: {num_clusters}")

    # Save images into their respective cluster folders
    save_images_by_cluster(image_folder, image_names, clusters)

    # Optional: Plot dendrogram to visualize the clusters
    plt.figure(figsize=(10, 7))
    dendrogram(
        linked,
        labels=image_names,
        orientation="top",
        distance_sort="descending",
        show_leaf_counts=True,
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Image")
    plt.ylabel("Distance")
    plt.show()
