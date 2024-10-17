import cv2
import numpy as np
from sklearn.decomposition import PCA
import os
from intersections import find_intersections_via_hit_or_miss
import threading

counter = 0
counter_lock = threading.Lock()

def preprocess_image(image, preprocess_image_args: list, *debug_mode):
    # Validate the input
    if len(preprocess_image_args) != 9:
        raise ValueError
    
    # Seperate the input arguments
    nlm_h, nlm_tws, nlm_sws, gb_kernel, clahe_cl, clahe_tgs, thresh_bs, thresh_c, morphx_kernel = preprocess_image_args

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert white pixels to gray
    h, w = gray.shape[:2]
    pixels = []
    # Iterates every pixel from the top-left pixel
    for x in range(w):
        for y in reversed(range(h)):
            if 245 <= gray[y, x] <= 255:
                # Returns the first white pixel
                gray[y, x] = 190

    # Apply Non-Local Means Denoising with a lower h value to preserve more detail
    denoised = cv2.fastNlMeansDenoising(gray, None, h=nlm_h, templateWindowSize=nlm_tws, searchWindowSize=nlm_sws)

    # Apply Gaussian blur with a smaller kernel to reduce noise but preserve more details
    blurred = cv2.GaussianBlur(denoised, gb_kernel, 0)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clahe_cl, tileGridSize=clahe_tgs)
    enhanced_gray = clahe.apply(blurred)

    # Use adaptive thresholding for local binarization
    thresh = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh_bs, thresh_c)

    # Apply morphological closing to ensure veins are connected, but use a smaller kernel to avoid over-smoothing
    kernel_close = np.ones(morphx_kernel, np.uint8)
    closed_binary = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    # Remove small pieces with an area less than 1000 pixels
    labelnum, labelimg, stats, centroids = cv2.connectedComponentsWithStats(closed_binary)
    mask = np.zeros_like(closed_binary)
    for label in range(1, labelnum):
        x, y, w, h, size = stats[label]
        if size <= 1000:
            contour = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]])
            cv2.drawContours(mask, [contour], -1, 255, -1)
    cleaned_binary = cv2.bitwise_and(closed_binary, ~mask)

    scope_vars = vars()
    
    for item in debug_mode:
        try:
            cv2.imshow(item, scope_vars[item])
        except Exception as e:
            pass

    return cleaned_binary


def remove_noise(binary_image, remove_noise_args: list, *debug_mode):
    # Validate the input
    if len(remove_noise_args) != 2:
        raise ValueError
    
    # Seperate the input arguments
    kernel_o, kernel_c = remove_noise_args

    # Apply morphological operations to remove small noise
    kernel_open = np.ones(kernel_o, np.uint8)
    kernel_close = np.ones(kernel_c, np.uint8)

    # Opening to remove small noise
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)

    # Closing to close small gaps in the veins
    cleaned_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel_close)

    scope_vars = vars()
    
    for item in debug_mode:
        try:
            cv2.imshow(item, scope_vars[item])
        except Exception as e:
            pass

    return cleaned_image


def remove_small_black_regions(binary_image, *debug_mode):
    # Find connected components with an area less than 1000 pixels
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - binary_image, connectivity=8)

    # Create an output image (initialized as white)
    output_image = np.ones_like(binary_image) * 255

    # Iterate through each component found
    for i in range(1, num_labels):  # skip the background
        # Check if the component's area is less than 300 pixels
        if stats[i, cv2.CC_STAT_AREA] < 300:
            # Turn the small black object into white
            output_image[labels == i] = 255
        else:
            # Keep the larger black components as is
            output_image[labels == i] = 0

    scope_vars = vars()
    
    for item in debug_mode:
        try:
            cv2.imshow(item, scope_vars[item])
        except Exception as e:
            pass

    return output_image


def find_orientation(binary_image):
    # Extract coordinates of non-zero pixels (vein points)
    points = np.column_stack(np.where(binary_image > 0))  # Get non-zero coordinates

    # Apply PCA on the points
    pca = PCA(n_components=2)
    pca.fit(points)

    # Get the first principal component (the direction of maximum variance)
    # The angle of the first principal component represents the orientation
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])

    # Convert from radians to degrees
    angle = np.degrees(angle)

    return angle


def rotate_image(image, angle, *debug_mode):
    # Get image dimensions
    (h, w) = image.shape[:2]

    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Perform rotation
    M = cv2.getRotationMatrix2D(center, -(angle - 90), 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Add borders
    padded_rotated = cv2.copyMakeBorder(rotated, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    scope_vars = vars()
    
    for item in debug_mode:
        try:
            cv2.imshow(item, scope_vars[item])
        except Exception as e:
            pass
    
    return padded_rotated


def extract_skeleton(binary_image, *debug_mode):
    skeleton = cv2.ximgproc.thinning(binary_image, cv2.ximgproc.THINNING_ZHANGSUEN)

    scope_vars = vars()
    
    for item in debug_mode:
        try:
            cv2.imshow(item, scope_vars[item])
        except Exception as e:
            pass

    return skeleton
    


def crop_image(skeletonized, image_path, *debug_mode):
    # Helper function for cropping
    def find_the_leftmost_pixel(image):
        h, w = image.shape[:2]
        pixels = []
        # Iterates every pixel from the top-left pixel
        for x in range(w):
            for y in reversed(range(h)):
                if image[y, x] == 255:
                    # Returns the first white pixel
                    return x
                               
    def upper_border_finder(image):    
        if image is None:
            print("Error loading image.")
            return None
        
        h, w = image.shape
        # Adjust width for margin on the right
        w -= 20
        white_pixels = []    
        # Iterate through every pixels
        for height in range(h):
            white_pixel_count = 0
            for width in range(20, w):
                pixel = image[height, width]
                if pixel == 255:
                    white_pixel_count += 1
            # Append a tuple that represents the number of white pixels in each row
            white_pixels.append((height, white_pixel_count))
        
        # Create a list that holds the borders between white lines and black areas
        borders = []
        
        for i in range(1, len(white_pixels)):
            upper_line = white_pixels[i]
            lower_line = white_pixels[i-1]

            # Append the upper line as a border if it is a valid line
            if upper_line[1] > 1 and lower_line[1] == 0 and upper_line[1] - lower_line[1] > 5:
                borders.append(upper_line)

        # The only valid state for our case is if there are two elements
        if len(borders) == 2:
            return borders[1][0]
        else:
            return None
    h, w = skeletonized.shape
    if "AT" in image_path:
        skeletonized = skeletonized[:, :-150] 
    if "RO" in image_path:
        skeletonized = skeletonized[:, :-150]  
    if "HU" in image_path and "2019" not in  image_path:
        skeletonized = skeletonized[:, :-170 ]
    if "HU" in image_path and "2019" in image_path:
        skeletonized = skeletonized[:, :-100]
    if "MD" in image_path:
        skeletonized = skeletonized[:, :-100]
    if "PL" in image_path and (1000 < w):
        skeletonized = skeletonized[:, :-400]
    elif "PL" in image_path and (800 < w < 1000):
        skeletonized = skeletonized[:, :-200]
    elif "PL" in image_path and (600 < w < 800):
        skeletonized = skeletonized[:, :-10]
    # Find every intersection
    intersection_coords = find_intersections_via_hit_or_miss(skeletonized)
    
    h, w = skeletonized.shape[:2]
    average = 0
    if "PL" in image_path and (1000 < w):
        average = (w*35) / 100
    if "PL" in image_path and (800 < w < 1000):
        average = (w*25) / 100
    if "PL" in image_path and (600 < w , 800):
        average = (w*15) / 100
    left_intersection = w
    upper_intersection = h
    right_intersection = 0
    lower_intersection = 0


    # Find every border
    for coord in intersection_coords:
        if  coord[1] < left_intersection:
            left_intersection = coord[1]

        if  coord[0] < upper_intersection:
            upper_intersection = coord[0]

        if coord[1] > right_intersection:
            if coord[1] < w+20:
                right_intersection = coord[1]

        if coord[0] > lower_intersection:
            if coord[0] < w-20:
                lower_intersection = coord[0]

    upper_border = upper_border_finder(skeletonized)

    if  upper_border != None:
        upper_intersection = upper_border
    
    if left_intersection > average:
        left_intersection = find_the_leftmost_pixel(skeletonized)



    x1 = left_intersection-15
    x2 = right_intersection+15
    y1 = upper_intersection-25
    y2 = lower_intersection+15

    # Check if the border coordinates are out of bounds
    if x1 < 0:
        pad_left = -x1
        x1 = 0
    else:
        pad_left = 0

    if y1 < 0:
        pad_top = -y1
        y1 = 0
    else:
        pad_top = 0

    if x2 > skeletonized.shape[1]:
        pad_right = x2 - skeletonized.shape[1]
    else:
        pad_right = 0

    if y2 > skeletonized.shape[0]:
        pad_bottom = y2 - skeletonized.shape[0]
    else:
        pad_bottom = 0

    # Pad the image with black pixels if necessary
    padded_image = cv2.copyMakeBorder(skeletonized, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    cropped_image = padded_image[y1:y2, x1:x2]

    # Debugging option
    scope_vars = vars()
    
    for item in debug_mode:
        try:
            cv2.imshow(item, scope_vars[item])
        except Exception as e:
            pass

    return cropped_image


def process_bee_wing(image_path, args: list, out, *debug_mode):
    # Validate the input
    if len(args) != 11:
        raise ValueError
    
    # Seperate the arguments
    preprocess_image_args = args[0:9]
    remove_noise_args = args[9:11]

    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image (grayscale, blur, threshold)
    binary = preprocess_image(image, preprocess_image_args, *debug_mode)

    # Remove noise
    cleaned = remove_noise(binary, remove_noise_args, *debug_mode)

    # Remove small black regions
    cleaned = remove_small_black_regions(cleaned, *debug_mode)

    # Find the orientation and rotate the image
    angle = find_orientation(cleaned)
    rotated = rotate_image(cleaned, angle, *debug_mode)

    # Extract the skeleton of the vein structure
    skeleton = extract_skeleton(rotated, *debug_mode)

    cropped = crop_image(skeleton, image_path, *debug_mode)

    # Wait for a key press
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()

    os.makedirs(out, exist_ok=True)
    out = os.path.join(out, os.path.basename(image_path))

    cv2.imwrite(out, cropped)

    global counter

    with counter_lock:
        counter += 1
        #print(f"Skeletonized {image_path} ---> {out} | {counter}")


def main():
    # Example usage:
    input_image_path = r'/home/delta/Documents/original_wings_labeled/SI-0020-386-100654-L.dw.png'
    output_image_path = r'/home/delta/Documents'

    # nlm_h, nlm_tws, nlm_sws, gb_kernel, clahe_cl, clahe_tgs, thresh_bs, thresh_c, morphx_kernel, kernel_open, kernel_close
    arg_list = [12, 29, 42, (3, 3), 2.2, (20, 20), 43, 11.9, (2, 2), (2, 2), (5, 5)]

    # Process the bee wing image
    skeleton_image = process_bee_wing(input_image_path, arg_list, output_image_path, "padded_image", "cropped_image")

    # gray, denoised, blurred, enhanced_gray, thresh, closed_binary
    # opened_image, cleaned_image
    # output_image
    # rotated
    # skeleton


if __name__ == "__main__":
    main()