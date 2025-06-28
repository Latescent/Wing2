import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from multiprocessing import Pool, cpu_count

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

# Set the image folder path
image_folder = "/home/delta/Documents/new_database/Mod-labeled-Clean"  # Replace with your image folder path

# Load and preprocess image data
image_vectors, image_names = load_images_as_vectors(image_folder)

# Optionally reduce dimensionality using PCA (optional for very large datasets)
pca = PCA(n_components=50)  # Reduce to 50 components
image_vectors_pca = pca.fit_transform(image_vectors)

# Standardize the features
scaler = StandardScaler()
image_vectors_scaled = scaler.fit_transform(image_vectors_pca)

# Perform Hierarchical Clustering using Ward’s method
linked = linkage(image_vectors_scaled, method='ward')

# Define the distance threshold or the number of clusters you want to generate
distance_threshold = 135  # Adjust this value based on your data (e.g., 100)
clusters = fcluster(linked, t=distance_threshold, criterion='distance')

# Count the number of unique clusters
num_clusters = len(np.unique(clusters))
print(f"Number of clusters identified: {num_clusters}")

# Save images into their respective cluster folders
save_images_by_cluster(image_folder, image_names, clusters)

# Optional: Plot dendrogram to visualize the clusters
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=image_names, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Image')
plt.ylabel('Distance')
plt.show()