"""
Performs hierarchical clustering on a directory of images.

This script processes a folder of images by converting them into feature vectors,
reducing their dimensionality with PCA, and then applying hierarchical
clustering to group similar images. The resulting clusters are saved into
separate directories, and a dendrogram can be displayed for visualization.
"""

import os
import shutil
import sys
import yaml

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.data_utils import load_images_as_vectors


def save_images_by_cluster(image_folder: str, image_names: list[str], clusters: np.ndarray):
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
    # --- Load Configuration from the root directory ---
    # The path is relative to this script's location
    config_path = os.path.join(os.path.dirname(__file__), '../../configs', 'config.yaml')
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get general paths and method-specific parameters
    image_folder = config["data_dir"]
    output_folder = config["output_dir"]
    params = config["hierarchical_clustering"]
    n_components = params["n_components"]
    distance_threshold = params["distance_threshold"]

    # Load and preprocess image data
    image_vectors, image_names = load_images_as_vectors(image_folder)

    # Optionally reduce dimensionality using PCA (optional for very large datasets)
    pca = PCA(n_components)  # Reduce to 50 components
    image_vectors_pca = pca.fit_transform(image_vectors)

    # Standardize the features
    scaler = StandardScaler()
    image_vectors_scaled = scaler.fit_transform(image_vectors_pca)

    # Perform Hierarchical Clustering using Ward’s method
    linked = linkage(image_vectors_scaled, method="ward")

    # Define the distance threshold or the number of clusters you want to generate
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
        distance_sort="descending", #type: ignore
        show_leaf_counts=True,
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Image")
    plt.ylabel("Distance")
    plt.show()
