import cv2
import numpy as np

def find_intersections_via_hit_or_miss(skeleton, show=False):
    """
    Detect intersection points in a skeletonized binary image using morphological hit-or-miss operation.

    Parameters:
    skeleton (np.array): Binary image of the skeletonized structure.

    Returns:
    intersections (np.array): Binary image showing intersection points.
    intersection_coords (list): List of (x, y) coordinates of intersection points.
    """
    # Define a list of 3x3 masks that represent possible intersection patterns
    kernel_1 = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=np.uint8)
    
    kernel_2 = np.array([[1, 0, 1],
                         [0, 1, 0],
                         [1, 0, 1]], dtype=np.uint8)
    
    kernel_3 = np.array([[1, 1, 1],
                         [0, 1, 0],
                         [0, 1, 0]], dtype=np.uint8)
    
    kernel_4 = np.array([[0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=np.uint8)
    
    kernel_5 = np.array([[1, 0, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=np.uint8)
    
    kernel_6 = np.array([[1, 0, 1],
                         [0, 1, 0],
                         [0, 1, 0]], dtype=np.uint8)
    
    kernel_7 = np.array([[0, 1, 0],
                         [1, 1, 0],
                         [0, 0, 1]], dtype=np.uint8)
    
    kernel_8 = np.array([[1, 1, 1],
                         [0, 1, 0],
                         [0, 0, 1]], dtype=np.uint8)
    
    kernel_9 = np.array([[1, 1, 1],
                         [0, 1, 0],
                         [1, 0, 0]], dtype=np.uint8)
    
    kernel_10 = np.array([[1, 0, 1],
                          [0, 1, 0],
                          [1, 0, 0]], dtype=np.uint8)

    masks = [
        kernel_1,
        kernel_2,
        kernel_3, np.rot90(kernel_3, k=1), np.rot90(kernel_3, k=2), np.rot90(kernel_3, k=3),
        kernel_4, np.rot90(kernel_4, k=1), np.rot90(kernel_4, k=2), np.rot90(kernel_4, k=3),
        kernel_5, np.rot90(kernel_5, k=1), np.rot90(kernel_5, k=2), np.rot90(kernel_5, k=3),
        kernel_6, np.rot90(kernel_6, k=1), np.rot90(kernel_6, k=2), np.rot90(kernel_6, k=3),
        kernel_7, np.rot90(kernel_7, k=1), np.rot90(kernel_7, k=2), np.rot90(kernel_7, k=3),
        kernel_8, np.rot90(kernel_8, k=1), np.rot90(kernel_8, k=2), np.rot90(kernel_8, k=3),
        kernel_9, np.rot90(kernel_9, k=1), np.rot90(kernel_9, k=2), np.rot90(kernel_9, k=3),
        kernel_10, np.rot90(kernel_10, k=1), np.rot90(kernel_10, k=2), np.rot90(kernel_10, k=3)
    ]
    
    intersections = np.zeros_like(skeleton, dtype=np.uint8)

    # Apply hit-or-miss transform using each mask
    for mask in masks:
        hit_or_miss = cv2.morphologyEx(skeleton, cv2.MORPH_HITMISS, mask)
        intersections = np.logical_or(intersections, hit_or_miss).astype(np.uint8)
    
    # Get the coordinates of the intersection points in a standard format(list of tuples)
    x_list = list(np.where(intersections > 0)[0])
    y_list = list(np.where(intersections > 0)[1])

    for index in range(len(x_list)):
        x_list[index] = int(x_list[index])

    for index in range(len(y_list)):
        y_list[index] = int(y_list[index])

    coordination_list = list()

    for index in range(len(x_list)):
        coordination_list.append((x_list[index], y_list[index]))

    # Display the intersection points on the skeleton image
    display_image = cv2.cvtColor(skeleton * 255, cv2.COLOR_GRAY2BGR)
    for coord in coordination_list:
        cv2.circle(display_image, tuple(coord[::-1]), radius=3, color=(0, 0, 255), thickness=-1)

    # Show the result
    if show:
        cv2.imshow('Intersections', display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return coordination_list

#################################################################

def find_the_leftmost_pixel(image):
    h, w = image.shape[:2]
    pixels = []
    for x in range(w):
        for y in reversed(range(h)):
            if image[y, x] == 255:
                return x


def crop_image_with_opencv(image_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error loading image.")
        return None, None

    h, w = image.shape
    w -= 20  # Adjust width for margin on the right

    w_pixels = []
    
    # Iterate through the image pixels
    for x in range(h):
        number_white = 0
        for y in range(20, w):
            pixel = image[x, y]
            if pixel == 255:  # White pixel in grayscale image
                number_white += 1
        w_pixels.append((x, number_white))
    
    crop = []
    max_up = w_pixels[0]
    
    for i in range(1, len(w_pixels)):
        pix2 = w_pixels[i]
        pix1 = w_pixels[i-1]

        if pix2[1] > 1 and pix1[1] == 0 and pix2[1] - pix1[1] > 5:
            crop.append(pix2)

    if len(crop) == 2:
        return crop[1][0], max_up[0]
    else:
        return None, max_up[0]
    

def crop_image(skeleton, image_path):

    if "AT" or "RO" in image_path:
        skeleton = skeleton[:, :-150]
    if "HU" in image_path and "2019" not in image_path:
        skeleton = skeleton[:, :-60]
        
    _, skeleton_binary = cv2.threshold(skeleton, 127, 1, cv2.THRESH_BINARY)
    intersection_coords = find_intersections_via_hit_or_miss(skeleton_binary)
    print(len(intersection_coords))
    h, w = skeleton.shape[:2]

    avarage = (w*25) / 100
    left = w
    up = h
    for coord in intersection_coords:
        if  coord[1] < left:
            left = coord[1]
        if  coord[0] < up:
            up = coord[0]       

    right = 0
    down = 0
    for coord in intersection_coords:
        if coord[1] > right:
            if coord[1] < w-20:
                right = coord[1]
        if coord[0] > down:
            if coord[0] < w-20:
                down = coord[0]

    up_test = crop_image_with_opencv(image_path)
    if  up_test[0] != None:
        up = up_test[0]
        print(up_test)
    # else:
    #     print(up_test[1]) 
    #     up =  up_test[1]
    print(up)
    if left > avarage:
        left = find_the_leftmost_pixel(skeleton)

    point1 = (up-15, left-5)  
    point2 = (down+10, right+10)          
    x1 = point1[1]
    x2 = point2[1]
    y1 = point1[0]
    y2 = point2[0]
    
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    cropped_image = skeleton[y1:y2, x1:x2]
    cv2.imshow('crop', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cropped_image

img = cv2.imread("1.png", cv2.IMREAD_GRAYSCALE)
crop_image(img, "1.png")