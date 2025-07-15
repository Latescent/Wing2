import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
from multiprocessing import Pool, cpu_count
from PIL import Image
import shutil


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


def load_images_as_vectors(image_folder: str, image_size: tuple[int, int] = (512, 256)):
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
    # Set the image folder path
    image_folder = (
        "/home/neutral/Desktop/CODE/Mod-labeled"  # Replace with your image folder path
    )

    # Load and preprocess image data
    image_vectors, image_names = load_images_as_vectors(image_folder)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(image_vectors)

    # Apply DBSCAN
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    dbscan = DBSCAN(eps=15, min_samples=5)  # Note: eps may need significant tuning
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # Save the clustered images into folders
    save_images_by_cluster(image_folder, image_names, dbscan_labels)
    print(
        f"Clustering complete. Found {len(set(dbscan_labels) - {-1})} clusters and noise points."
    )
    print("Images have been saved to the 'output_clusters' directory.")

    # Dimensionality Reduction (for visualization)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plotting the DBSCAN clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=dbscan_labels,
        palette="viridis",
        legend="full",
        s=50,
    )
    plt.title("DBSCAN Clustering of Images (PCA-reduced)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster ID")
    plt.show()
