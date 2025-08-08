import cv2
import numpy as np
import argparse


def cleanup_intersections(
    coords: list[tuple[int, int]], distance_threshold: int = 5
) -> list[tuple[int, int]]:
    """
    Cleans up a list of coordinates by merging points that are close to each other.

    This function implements a simple clustering algorithm. It iterates through the
    points, and for each point, it forms a cluster of all other points within
    the specified distance threshold. The centroid of this cluster then replaces
    the individual points. This is more efficient than a pairwise comparison,
    especially for a large number of coordinates.

    Args:
        coords (list[tuple[int, int]]): A list of (y, x) coordinates to be cleaned.
        distance_threshold (int, optional): The maximum distance between points
            to be considered part of the same cluster. Defaults to 8.

    Returns:
        list[tuple[int, int]]: A list of merged (y, x) coordinates.
    """
    if not coords:
        return []

    # Use a copy of the list to safely remove items while iterating
    remaining_coords = coords[:]
    merged_coords = []

    while remaining_coords:
        # Start a new cluster with the first point in the list
        base_pt = remaining_coords.pop(0)
        cluster = [base_pt]

        # Find all other points within the threshold distance of the base point
        # A list comprehension is used for a concise inner loop
        other_pts_in_cluster = [
            pt
            for pt in remaining_coords
            if np.linalg.norm(np.array(base_pt) - np.array(pt)) < distance_threshold
        ]

        # Add found points to the cluster and remove them from the remaining list
        cluster.extend(other_pts_in_cluster)
        for pt in other_pts_in_cluster:
            remaining_coords.remove(pt)

        # Calculate the centroid of the cluster and add it to the results
        centroid = np.mean(cluster, axis=0)
        merged_coords.append(tuple(np.round(centroid).astype(int)))

    return merged_coords


def find_intersections(skeleton, show=False):
    """
    Detects intersection points in a skeletonized binary image using the
    morphological hit-or-miss transform.

    The hit-or-miss transform finds pixels in the image that match a specific
    pattern defined by a kernel. This function uses a set of kernels that
    represent various types of line intersections (e.g., T-junctions,
    X-junctions, Y-junctions).

    Args:
        skeleton (np.ndarray): A binary image (with values 0 and 1) representing
            the skeletonized structure.
        show (bool, optional): If True, the function will display the original
            skeleton with the detected intersections marked. Defaults to False.

    Returns:
        list[tuple[int, int]]: A cleaned list of (y, x) coordinates for each
            detected intersection point.
    """
    # Define all unique base kernels from the original script for full coverage.
    k_x = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    k_x_diag = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)

    # All other unique kernels that will be rotated to cover all orientations.
    # This list is now a comprehensive representation of the original kernels.
    kernels_to_rotate = [
        np.array(
            [[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=np.uint8
        ),  # Original kernel_6
        np.array(
            [[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=np.uint8
        ),  # Original kernel_3
        np.array(
            [[0, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
        ),  # Original kernel_4
        np.array(
            [[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
        ),  # Original kernel_5
        np.array(
            [[0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.uint8
        ),  # Original kernel_7
        np.array(
            [[1, 1, 1], [0, 1, 0], [0, 0, 1]], dtype=np.uint8
        ),  # Original kernel_8
        np.array(
            [[1, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8
        ),  # Original kernel_9
        np.array(
            [[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8
        ),  # Original kernel_10
    ]

    all_kernels = [k_x, k_x_diag]
    for kernel in kernels_to_rotate:
        for k in range(4):
            all_kernels.append(np.rot90(kernel, k=k))

    intersections = np.zeros_like(skeleton, dtype=np.uint8)
    for kernel in all_kernels:
        hit_or_miss = cv2.morphologyEx(skeleton, cv2.MORPH_HITMISS, kernel)
        intersections = np.logical_or(intersections, hit_or_miss)

    raw_coords = [tuple(coord) for coord in np.argwhere(intersections > 0)]
    cleaned_coords = cleanup_intersections(raw_coords)
    final_coords = []
    height = len(skeleton) - 1
    width = len(skeleton[0]) - 1
    print(height, width)
    for x, y in cleaned_coords:
        print(x, y)
        if x == 0 or y == 0 or x == height or y == width:
            continue
        final_coords.append((x, y))

    if show:
        display_image = cv2.cvtColor(skeleton * 255, cv2.COLOR_GRAY2BGR)
        for coord in final_coords:
            cv2.circle(
                display_image, coord[::-1], radius=5, color=(0, 0, 255), thickness=-1
            )
        cv2.imshow("Detected Intersections", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_coords


# Main execution block
if __name__ == "__main__":
    # Set up argument parser to make the script a command-line tool
    parser = argparse.ArgumentParser(
        description="Find and display intersections in a skeletonized image."
    )
    parser.add_argument(
        "image_path", type=str, help="Path to the input skeletonized image."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the image with detected intersections.",
    )

    args = parser.parse_args()

    # Load the image in grayscale
    try:
        skeleton_image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
        if skeleton_image is None:
            raise FileNotFoundError(f"Image not found at path: {args.image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        exit()

    # Ensure the image is binary (0s and 1s) for the morphological operations.
    # This assumes the skeleton is white (255) on a black (0) background.
    _, skeleton_binary = cv2.threshold(skeleton_image, 127, 1, cv2.THRESH_BINARY)

    # Delete the pixel wide border of the image
    h, w = skeleton_binary.shape
    skeleton_binary[0, :] = 0  # Top edge
    skeleton_binary[h - 1, :] = 0  # Bottom edge
    skeleton_binary[:, 0] = 0  # Left edge
    skeleton_binary[:, w - 1] = 0  # Right edge

    # Find and optionally display intersections
    intersection_coords = find_intersections(skeleton_binary, show=args.show)

    print(f"Found {len(intersection_coords)} intersection(s).")
    print("Intersection Coordinates (y, x):", intersection_coords)
