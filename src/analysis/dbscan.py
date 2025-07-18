import os
import shutil
import sys
import yaml

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.data_utils import load_images_as_vectors


def save_images_by_cluster(
    image_folder: str, image_names: list[str], clusters: np.ndarray
):
    """Saves images into separate folders based on their cluster ID.

    Args:
        image_folder (str): The original directory where the images are stored.
        image_names (list[str]): An ordered list of image filenames.
        clusters (np.ndarray): An array of cluster IDs corresponding to each image.
    """
    output_dir = "output_clusters"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, cluster_id in enumerate(clusters):
        cluster_folder = os.path.join(output_dir, f"cluster_{cluster_id}")
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)

        src_path = os.path.join(image_folder, image_names[i])
        dest_path = os.path.join(cluster_folder, image_names[i])
        shutil.copy(src_path, dest_path)


if __name__ == "__main__":
    # --- Load Configuration ---
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/', 'config.yaml')
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get general paths and method-specific parameters
    image_folder = config["data_dir"]
    output_folder = config["output_dir"]
    params = config["dbscan_clustering"]
    eps = params["eps"]
    min_samples = params["min_samples"]

    # Load and preprocess image data
    image_vectors, image_names = load_images_as_vectors(image_folder)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(image_vectors)

    # Apply DBSCAN
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Note: eps may need significant tuning
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # Save the clustered images into folders
    save_images_by_cluster(image_folder, image_names, dbscan_labels)
    print(
        f"Clustering complete. Found {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)} clusters and noise points."
    )
    print("Images have been saved to the 'output_clusters' directory.")

    # Dimensionality Reduction (for visualization)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plotting the DBSCAN clusters
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', s=50)
    plt.title("DBSCAN Clustering of Images (PCA-reduced)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster ID")
    plt.show()
