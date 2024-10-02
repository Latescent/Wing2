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

# Example synthetic data (replace with actual feature data)
# np.random.seed(42)
# features = np.random.rand(100, 50)

# Function to load and preprocess a single image
def process_single_image(filename):
    print(f"Loading image {filename}")
    img = Image.open(filename).convert('L')  # Convert to grayscale
    img = img.resize((128, 128))  # Resize for consistency
    img_array = np.array(img).flatten()  # Flatten the image into a vector
    return img_array

# Function to load images in parallel and convert to feature vectors
def load_images_as_vectors(image_folder, image_size=(512, 256)):
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")]
    image_names = [os.path.basename(path) for path in image_paths]
    image_paths.sort()
    image_names.sort()
    
    # Use multiprocessing to load images in parallel
    with Pool(cpu_count()) as pool:
        image_vectors = pool.map(process_single_image, image_paths)
    
    return np.array(image_vectors), image_names

# Function to create directories for each cluster and move images into their respective folders
def save_images_by_cluster(image_folder, image_names, clusters):
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

# Set the image folder path
image_folder = "/home/neutral/Desktop/CODE/Mod-labeled"  # Replace with your image folder path

# Load and preprocess image data
image_vectors, image_names = load_images_as_vectors(image_folder)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(image_vectors)

# Dimensionality Reduction (for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Plotting the DBSCAN clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dbscan_labels, palette='viridis', legend='full')
plt.title("DBSCAN Clustering")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()