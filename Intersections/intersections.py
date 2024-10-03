import cv2
import numpy as np

def remove_repetitives(data: tuple):
    def compare(lst1, lst2):
        if abs(lst1[0] - lst2[0]) < 4:
            if abs(lst1[1] - lst2[1]) < 4:
                temp_tup = (int((lst1[0] + lst2[0]) / 2), int((lst1[1] + lst2[1]) / 2))
                return temp_tup
        return None
                
    for index1 in range(len(data) - 1):
        for index2 in range(index1 + 1, len(data) - 1):
            try:
                if result := compare(data[index1], data[index2]):
                    data[index2] = tuple()
                    data[index1] = result
            except Exception:
                pass
    
    while tuple() in data:
        data.remove(tuple())

    return data

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

    coordination_list = remove_repetitives(coordination_list)

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

# Example usage
if __name__ == "__main__":
    # Load a skeletonized binary image (assuming it's already processed)
    skeleton = cv2.imread('/home/neutral/Documents/Wings/modified_wings_labeled/AT-0001-031-003686-R.dw.png', cv2.IMREAD_GRAYSCALE)

    # Ensure it's binary (0 and 1)
    _, skeleton_binary = cv2.threshold(skeleton, 127, 1, cv2.THRESH_BINARY)

    # Find and display intersections
    intersection_coords = find_intersections_via_hit_or_miss(skeleton_binary, show=True)
    print("Intersection Coordinates:", intersection_coords)