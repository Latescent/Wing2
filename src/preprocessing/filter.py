"""
This module provides a pipeline for processing bee wing images to extract and isolate the vein structure.

The main workflow involves:
1.  Loading an image of a bee wing.
2.  Preprocessing the image through a series of computer vision techniques including
    denoising, contrast enhancement, and binarization.
3.  Cleaning the binary image to remove noise and small artifacts.
4.  Determining the orientation of the wing veins and rotating the image to a
    standardized alignment.
5.  Extracting the skeleton of the vein network.
6.  Cropping the image to the relevant region of interest containing the veins.
7.  Saving the final processed image.

The primary function for executing this pipeline is `process_bee_wing`.
"""
import cv2
import numpy as np
from sklearn.decomposition import PCA
import os
from intersections import find_intersections_via_hit_or_miss
import threading

counter = 0
counter_lock = threading.Lock()


def show_debug_images(scope, names_to_show):
    """Displays intermediate images for debugging purposes.

    Args:
        scope (dict): A dictionary of local variables from the caller's scope,
                      typically obtained by calling `locals()`.
        names_to_show (tuple): A tuple of strings, where each string is the
                               variable name of an image to be displayed.
    """
    if not names_to_show:
        return

    for name in names_to_show:
        image = scope.get(name)
        if image is not None:
            try:
                cv2.imshow(name, image)
            except cv2.error:
                print(f"Warning: Could not display debug image '{name}'. "
                      f"It may not be a valid image format for cv2.imshow().")


def preprocess_image(image, preprocess_image_args: list, *debug_mode):
    """Applies a series of preprocessing steps to a bee wing image.

    This function converts the image to grayscale, reduces noise, enhances
    contrast, applies adaptive thresholding, and cleans the resulting binary
    image to prepare it for further analysis.

    Args:
        image (np.ndarray): The input image in BGR format.
        preprocess_image_args (list): A list of 9 parameters for image
            processing functions.
        *debug_mode (str): Optional strings specifying which intermediate images
            to display for debugging purposes (e.g., 'gray', 'denoised').

    Returns:
        np.ndarray: The preprocessed binary image.

    Raises:
        ValueError: If `preprocess_image_args` does not contain exactly 9 elements.
    """
    if len(preprocess_image_args) != 9:
        raise ValueError

    nlm_h, nlm_tws, nlm_sws, gb_kernel, clahe_cl, clahe_tgs, thresh_bs, \
        thresh_c, morphx_kernel = preprocess_image_args

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    for x in range(w):
        for y in reversed(range(h)):
            if 245 <= gray[y, x] <= 255:
                gray[y, x] = 190

    denoised = cv2.fastNlMeansDenoising(
        gray, None, h=nlm_h, templateWindowSize=nlm_tws,
        searchWindowSize=nlm_sws
    )
    blurred = cv2.GaussianBlur(denoised, gb_kernel, 0)
    clahe = cv2.createCLAHE(clipLimit=clahe_cl, tileGridSize=clahe_tgs)
    enhanced_gray = clahe.apply(blurred)
    thresh = cv2.adaptiveThreshold(
        enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, thresh_bs, thresh_c
    )
    kernel_close = np.ones(morphx_kernel, np.uint8)
    closed_binary = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    labelnum, labelimg, stats, centroids = cv2.connectedComponentsWithStats(
        closed_binary
    )
    mask = np.zeros_like(closed_binary)
    for label in range(1, labelnum):
        x, y, w, h, size = stats[label]
        if size <= 1000:
            contour = np.array(
                [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]]
            )
            cv2.drawContours(mask, [contour], -1, 255, -1)
    cleaned_binary = cv2.bitwise_and(closed_binary, ~mask)

    show_debug_images(locals(), debug_mode)

    return cleaned_binary


def remove_noise(binary_image, remove_noise_args: list, *debug_mode):
    """Removes noise from a binary image using morphological operations.

    Args:
        binary_image (np.ndarray): The input binary image.
        remove_noise_args (list): A list of 2 kernel sizes for morphological
            operations.
        *debug_mode (str): Optional strings specifying which intermediate images
            to display (e.g., 'opened_image').

    Returns:
        np.ndarray: The cleaned binary image.

    Raises:
        ValueError: If `remove_noise_args` does not contain exactly 2 elements.
    """
    if len(remove_noise_args) != 2:
        raise ValueError

    kernel_o, kernel_c = remove_noise_args
    kernel_open = np.ones(kernel_o, np.uint8)
    kernel_close = np.ones(kernel_c, np.uint8)
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)
    cleaned_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE,
                                     kernel_close)

    show_debug_images(locals(), debug_mode)

    return cleaned_image


def remove_small_black_regions(binary_image, *debug_mode):
    """Removes small, isolated black regions (holes) from a binary image.

    It identifies all connected components of black pixels and fills those
    with an area smaller than a predefined threshold (300 pixels).

    Args:
        binary_image (np.ndarray): The input binary image where veins are black.
        *debug_mode (str): Optional strings specifying which intermediate images
            to display (e.g., 'output_image').

    Returns:
        np.ndarray: A binary image with small black regions removed.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        255 - binary_image, connectivity=8
    )
    output_image = np.ones_like(binary_image) * 255
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 300:
            output_image[labels == i] = 255
        else:
            output_image[labels == i] = 0

    show_debug_images(locals(), debug_mode)

    return output_image


def find_orientation(binary_image):
    """Calculates the dominant orientation of veins in the binary image.

    Uses Principal Component Analysis (PCA) on the coordinates of the vein
    pixels to determine the primary axis of variance, which corresponds to the
    overall orientation of the wing veins.

    Args:
        binary_image (np.ndarray): The binary image of the wing veins.

    Returns:
        float: The orientation angle in degrees.
    """
    points = np.column_stack(np.where(binary_image > 0))
    pca = PCA(n_components=2)
    pca.fit(points)
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    angle = np.degrees(angle)
    return angle


def rotate_image(image, angle, *debug_mode):
    """Rotates the image to align the vein structure vertically.

    The image is rotated to counteract the angle found by `find_orientation`,
    aiming for a 90-degree (vertical) alignment. A border is added to
    prevent loss of data near the edges after rotation.

    Args:
        image (np.ndarray): The image to be rotated.
        angle (float): The angle (in degrees) to rotate the image.
        *debug_mode (str): Optional strings specifying which intermediate images
            to display (e.g., 'rotated').

    Returns:
        np.ndarray: The rotated and padded image.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -(angle - 90), 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    padded_rotated = cv2.copyMakeBorder(
        rotated, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    show_debug_images(locals(), debug_mode)

    return padded_rotated


def extract_skeleton(binary_image, *debug_mode):
    """Extracts the skeleton of the vein structure from a binary image.

    This function uses a thinning algorithm to reduce the veins to a
    one-pixel-wide skeleton.

    Args:
        binary_image (np.ndarray): The binary input image of the veins.
        *debug_mode (str): Optional string to display the skeleton image.

    Returns:
        np.ndarray: The skeletonized image.
    """
    skeleton = cv2.ximgproc.thinning(binary_image,
                                     cv2.ximgproc.THINNING_ZHANGSUEN)

    show_debug_images(locals(), debug_mode)

    return skeleton


def crop_image(skeletonized, image_path, *debug_mode):
    """Crops the skeletonized image to the relevant vein area.

    This function first applies hard-coded adjustments based on the image's
    origin (from its filename). It then identifies the bounding box of the
    vein intersections and uses it to crop the image.

    Args:
        skeletonized (np.ndarray): The skeletonized image of the veins.
        image_path (str): The file path of the original image, used to
            determine origin-specific cropping rules.
        *debug_mode (str): Optional string to display the cropped image.

    Returns:
        np.ndarray: The cropped image.
    """
    def find_the_leftmost_pixel(image):
        h, w = image.shape[:2]
        for x in range(w):
            for y in reversed(range(h)):
                if image[y, x] == 255:
                    return x
        return 0

    def upper_border_finder(image):
        if image is None:
            return None
        h, w = image.shape
        w -= 20
        white_pixels = []
        for height in range(h):
            white_pixel_count = np.count_nonzero(image[height, 20:w] == 255)
            white_pixels.append((height, white_pixel_count))

        borders = []
        for i in range(1, len(white_pixels)):
            upper_line = white_pixels[i]
            lower_line = white_pixels[i - 1]
            if upper_line[1] > 1 and lower_line[1] == 0 and \
               upper_line[1] - lower_line[1] > 5:
                borders.append(upper_line)

        return borders[1][0] if len(borders) == 2 else None

    # Apply pre-cropping based on image origin from filename
    h_orig, w_orig = skeletonized.shape
    if "AT" in image_path:
        skeletonized = skeletonized[:, :-150]
    if "RO" in image_path:
        skeletonized = skeletonized[:, :-150]
    if "HU" in image_path and "2019" not in image_path:
        skeletonized = skeletonized[:, :-170]
    if "HU" in image_path and "2019" in image_path:
        skeletonized = skeletonized[:, :-100]
    if "MD" in image_path:
        skeletonized = skeletonized[:, :-100]
    if "PL" in image_path and (1000 < w_orig):
        skeletonized = skeletonized[:, :-400]
    elif "PL" in image_path and (800 < w_orig < 1000):
        skeletonized = skeletonized[:, :-200]
    elif "PL" in image_path and (600 < w_orig < 800):
        skeletonized = skeletonized[:, :-10]

    intersection_coords = find_intersections_via_hit_or_miss(skeletonized)
    h, w = skeletonized.shape[:2]
    average = 0
    if "PL" in image_path:
        if 1000 < w: average = (w * 35) / 100
        elif 800 < w < 1000: average = (w * 25) / 100
        elif 600 < w < 800: average = (w * 15) / 100

    left_intersection, upper_intersection = w, h
    right_intersection, lower_intersection = 0, 0

    if intersection_coords:
        coords_arr = np.array(intersection_coords)
        left_intersection = np.min(coords_arr[:, 1])
        upper_intersection = np.min(coords_arr[:, 0])
        right_intersection = np.max(coords_arr[coords_arr[:, 1] < w - 2, 1])
        lower_intersection = np.max(coords_arr[coords_arr[:, 0] < h - 20, 0])

    upper_border = upper_border_finder(skeletonized)
    if upper_border is not None:
        upper_intersection = upper_border
    if left_intersection > average and average > 0:
        left_intersection = find_the_leftmost_pixel(skeletonized)

    x1, x2 = int(left_intersection - 15), int(right_intersection + 15)
    y1, y2 = int(upper_intersection - 25), int(lower_intersection + 15)
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    cropped_image = skeletonized[y1:y2, x1:x2]

    show_debug_images(locals(), debug_mode)

    return cropped_image


def process_bee_wing(image_path, args: list, out, *debug_mode):
    """Processes a single bee wing image from loading to saving the skeleton.

    This function orchestrates the entire pipeline: loading, preprocessing,
    noise removal, orientation, rotation, skeletonization, and cropping.
    The final skeletonized image is saved to the specified output directory.

    Args:
        image_path (str): The path to the input bee wing image.
        args (list): A list of 11 parameters required for the processing
            functions.
        out (str): The path to the output directory where the processed
            image will be saved.
        *debug_mode (str): Optional strings to enable debug views in the
            called functions.

    Raises:
        ValueError: If `args` does not contain exactly 11 elements.
    """
    if len(args) != 11:
        raise ValueError

    preprocess_image_args = args[0:9]
    remove_noise_args = args[9:11]

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    binary = preprocess_image(image, preprocess_image_args, *debug_mode)
    cleaned = remove_noise(binary, remove_noise_args, *debug_mode)
    cleaned = remove_small_black_regions(cleaned, *debug_mode)
    angle = find_orientation(cleaned)
    rotated = rotate_image(cleaned, angle, *debug_mode)
    skeleton = extract_skeleton(rotated, *debug_mode)
    cropped = crop_image(skeleton, image_path, *debug_mode)

    if debug_mode:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    os.makedirs(out, exist_ok=True)
    out_path = os.path.join(out, os.path.basename(image_path))
    cv2.imwrite(out_path, cropped)

    global counter
    with counter_lock:
        counter += 1
        print(f"Skeletonized {image_path} ---> {out_path} | {counter}")


def main():
    """Main function to demonstrate the bee wing processing pipeline."""
    # Please update the input and output paths before running
    input_image_path = r'path/to/your/image.png'
    output_directory = r'path/to/your/output_directory'

    arg_list = [12, 29, 42, (3, 3), 2.2, (20, 20), 43, 11.9, (2, 2), (2, 2), (5, 5)]

    # To run with debug views, pass the names of intermediate images as strings
    process_bee_wing(
        input_image_path, arg_list, output_directory,
        'gray', 'denoised', 'blurred', 'enhanced_gray', 'thresh',
        'closed_binary', 'cleaned_image', 'rotated', 'skeleton', 'cropped'
    )


if __name__ == "__main__":
    main()
