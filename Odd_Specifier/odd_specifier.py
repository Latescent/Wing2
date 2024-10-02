import cv2, sys, os, concurrent.futures, numpy as np
from sklearn.ensemble import IsolationForest
from skimage import filters
from scipy.spatial import distance
sys.path.append(os.path.abspath("/home/neutral/Desktop/CODE/imgprcs/Intersections"))
from intersections import find_intersections_via_hit_or_miss

input_dir = "/home/neutral/Documents/Wings/modified_wings_labeled"
clean_dir = "/home/neutral/Documents/Wings/Mod-labeled-Clean"
noisy_dir = "/home/neutral/Documents/Wings/Mod-labeled-Noisy"

os.makedirs(clean_dir, exist_ok=True)
os.makedirs(noisy_dir, exist_ok=True)

################################################

def min_distance(coord, other_object):
    return min([distance.euclidean(coord, other_coord) for other_coord in other_object])

# Function to find the closest matches of coordinates between two objects
def compare_objects(object1, object2, threshold=1.0):
    matched_coords = 0
    for coord in object1:
        if any(min_distance(coord, object2) < threshold for coord in object1):
            matched_coords += 1
    return matched_coords

# Function to identify and exclude odd objects
def exclude_odd_objects(objects_list, match_threshold=0.8, distance_threshold=1.0):
    valid_objects = []
    for obj in objects_list:
        match_count = 0
        for other_obj in objects_list:
            if obj is not other_obj:
                # Compare each object with all others
                matched = compare_objects(obj, other_obj, distance_threshold)
                if matched / len(obj) >= match_threshold:
                    match_count += 1
        # Exclude object if not enough matches found
        if match_count >= len(objects_list) * match_threshold:
            valid_objects.append(obj)
    return valid_objects

################################################

def calculate_noise_score_contours(image):
    # Noise detection method based on contours. Useful in case of broken wings
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    noise_score = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            noise_score += 1
            
    return noise_score

def calculate_noise_score_edges(image):
    # Noise detection method based on edges. Useful in case of forks in wings and extra intersections
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    edges = filters.sobel(binary_image)
    std_edge = np.std(edges)
    return std_edge

def extract_features(image_path):
    # Utilize both methods and assign an attribute tuple to each image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noise_score_contours = calculate_noise_score_contours(image)
    noise_score_edge = calculate_noise_score_edges(image)
    return (noise_score_contours, noise_score_edge)

def extract_features_parallel(image_name):
    # Parallelize feature extraction
    image_path = os.path.join(input_dir, image_name)
    image_features = extract_features(image_path)
    print(f"Extracted features of {image_path}")
    return image_features, image_path

def classify_images():
    global input_dir, clean_dir, noisy_dir
    # Read every image
    images = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    images.sort()

    # Extract every feature of every image
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(extract_features_parallel, images))

    print("\nPreparing model")
    features = [result[0] for result in results]
    image_paths = [result[1] for result in results]
    print("\nModel prepared")
    features_array = np.array(features)
    print("\nTraining")

    iso_forest = IsolationForest(contamination=0.2, random_state=42, n_jobs=-1)
    predictions = iso_forest.fit_predict(features_array) # fit the Isolation Forest
    print("\nModel trained. Saving the images")
    # Classify images in two separate folders
    for i, image_path in enumerate(image_paths):
        if predictions[i] == 1:
            dest_path = os.path.join(clean_dir, os.path.basename(image_path))
        else:
            dest_path = os.path.join(noisy_dir, os.path.basename(image_path))
        
        cv2.imwrite(dest_path, cv2.imread(image_path))

    print("\nClassification complete. Images have been sorted into clean and noisy folders.")

classify_images()

# Rescale all images, odd intersections would significantly impact the average value of intersection. we can use the average value as an argument for iso-forest