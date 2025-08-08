"""
Converts a skeletonized image into a graph representation.

This script takes a binary skeleton image, identifies intersection points (nodes)
and the paths connecting them (edges), and constructs a `networkx` graph. It
also provides a function to visualize the resulting graph overlaid on the
original image.
"""

import cv2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from math import sqrt
from intersections import find_intersections


def find_neighbours(
    image: np.ndarray, coordinates: tuple[int, int], traversed: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Finds valid, untraversed neighboring white pixels for a given coordinate.

    Checks the 8-pixel neighborhood around a given (row, col) coordinate. A neighbor
    is considered valid if it is within the image bounds, is a white pixel (value 255),
    and has not been previously traversed.

    Args:
        image (numpy.ndarray): The binary image array.
        coordinates (tuple[int, int]): A (row, col) tuple for the pixel whose neighbors are to be found.
        traversed (set[tuple[int, int]]): A set of (row, col) tuples of already visited pixels.

    Returns:
        list: A list of (row, col) tuples representing the valid neighbors.
    """
    height = len(image)
    width = len(image[0])
    valid_neighbours = []

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue

            r, c = coordinates[0] + i, coordinates[1] + j

            if (
                0 <= r < height
                and 0 <= c < width
                and image[r][c] == 255
                and (r, c) not in traversed
            ):
                valid_neighbours.append((r, c))

    return valid_neighbours


def dist(coord1: tuple, coord2: tuple) -> float:
    """Calculates the Euclidean distance between two points.

    Args:
        coord1 (tuple): The (row, col) coordinates of the first point.
        coord2 (tuple): The (row, col) coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two coordinates.
    """
    return sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def trace_one_path_bfs(
    start_pixel: tuple[int, int],
    source: tuple[int, int],
    image: np.ndarray,
    nodes_list: list[tuple[int, int]],
    proximity_threshold: int = 5,
) -> tuple[int | None, list]:
    """Traces a path from a starting pixel until it reaches a node using BFS.

    This function performs a Breadth-First Search (BFS) starting from a single pixel
    to find a path that connects to any node in the `nodes_list`. The search terminates
    when the path gets within a specified pixel distance (`proximity_threshold`) of a
    destination node. The source node itself is ignored as a potential destination.

    Args:
        start_pixel (tuple[int, int]): The (row, col) coordinate to begin the trace from.
        source (tuple[int, int]): The (row, col) coordinate of the source node from which this
            path originates. This is used to prevent tracing back to the start.
        image (numpy.ndarray): The binary skeleton image.
        nodes_list (list[tuple[int, int]]): A list of (row, col) tuples representing all nodes (junctions).
        proximity_threshold (int, optional): The maximum distance in pixels to consider
            a pixel as "reaching" a node. Defaults to 5.

    Returns:
        tuple[int, list]: A tuple containing:
            - int: The index (ID) of the destination node found, or None if no node is reached.
            - list: A list of (row, col) tuples representing the pixels in the traced path.
              Empty if no path is found.
    """
    q = deque([(start_pixel, [start_pixel])])  # Queue: (current_pixel, path_taken)
    visited_in_this_path = {source, start_pixel}

    while q:
        current_pixel, path = q.popleft()

        for neighbor in find_neighbours(image, current_pixel, visited_in_this_path):
            if dist(source, neighbor) < 2:
                continue
            for node_id, node_coord in enumerate(nodes_list):
                if node_coord == source:
                    continue
                if dist(neighbor, node_coord) < proximity_threshold:
                    # Found a junction! Return its ID and the path.
                    return node_id, path + [neighbor]

            visited_in_this_path.add(neighbor)
            q.append((neighbor, path + [neighbor]))

    return None, []  # Path did not lead to a junction


def path_trace(image: np.ndarray, verbose: bool = False) -> tuple[nx.Graph, list, set]:
    """Constructs a graph representation from a skeletonized image.

    This function identifies all intersection points (nodes) in the image and then
    traces the paths (edges) between them. It systematically explores paths from each
    node, using BFS to find connections to other nodes. Each successfully traced path
    is added as an edge to a `networkx` graph.

    Args:
        image (numpy.ndarray): The binary skeleton image where paths are white (255).
        verbose (bool, optional): If True, prints progress messages. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - networkx.Graph: The resulting graph where nodes are junctions and edges
              are the paths between them.
            - list: A list of (row, col) coordinates for each node.
            - set: A set of all (row, col) pixel coordinates that form the traversed edges.
    """
    if verbose:
        print("Finding intersections...")
    nodes = find_intersections(image)
    if not nodes:
        print("No nodes found.")
        return nx.Graph(), [], set()

    G = nx.Graph()
    # Node IDs are their index in the 'nodes' list
    for i in range(len(nodes)):
        G.add_node(i)

    traversed_edge_pixels = set()

    for start_node_id, start_node_coord in enumerate(nodes):
        for neighbor in find_neighbours(image, start_node_coord, traversed_edge_pixels):
            if neighbor in traversed_edge_pixels:
                continue

            end_node_id, path_pixels = trace_one_path_bfs(
                neighbor, start_node_coord, image, nodes
            )

            if end_node_id is not None and path_pixels:
                if start_node_id != end_node_id:
                    G.add_edge(
                        start_node_id,
                        end_node_id,
                        length=dist(start_node_coord, nodes[end_node_id]),
                    )

                for pixel in path_pixels:
                    traversed_edge_pixels.add(pixel)

    return G, nodes, traversed_edge_pixels


def visualize_graph_on_image(
    graph: nx.Graph,
    image: np.ndarray,
    node_coordinates: list,
    traversed_pixels: set | None = None,
    verbose: bool = False,
) -> None:
    """Overlays a graph visualization on the source image.

    Draws the nodes and edges of the provided `networkx` graph on top of the
    original skeleton image. It can also optionally highlight all the pixels that
    were identified as part of an edge path in green.

    Args:
        graph (networkx.Graph): The graph to visualize.
        image (numpy.ndarray): The background skeleton image.
        node_coordinates (list): A list of (row, col) tuples for each node, indexed by node ID.
        traversed_pixels (set, optional): A set of (row, col) pixels to highlight.
            Defaults to None.
        verbose (bool, optional): If True, prints progress messages. Defaults to False.
    """
    if not node_coordinates:
        print("Cannot visualize: No node coordinates provided.")
        return
    if verbose:
        print("Generating visualization...")

    # Create a color version of the image to draw on
    display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # If traversed pixels are provided, color them green
    if traversed_pixels:
        if verbose:
            print(f"Highlighting {len(traversed_pixels)} traversed pixels in green...")
        for r, c in traversed_pixels:
            # Set the pixel at (row, col) to green (BGR format)
            display_image[r, c] = (0, 255, 0)

    pos = {i: (coord[1], coord[0]) for i, coord in enumerate(node_coordinates)}
    _, ax = plt.subplots(figsize=(12, 12))

    # Display the (potentially colored) image
    ax.imshow(display_image)

    # Draw the graph over the image
    nx.draw(
        graph,
        pos,
        ax=ax,
        with_labels=True,
        node_size=250,
        node_color="cyan",
        edge_color="magenta",
        width=2.5,
        font_size=10,
        font_color="black",
    )
    ax.set_title("Generated Graph on Skeleton Image", fontsize=16)
    plt.show()


if __name__ == "__main__":
    image_path = "/home/neutral/Documents/HR-0030-055-100478-R.dw_processed.png"

    # Load the image in grayscale
    try:
        skeleton_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if skeleton_image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        exit()

    G, nodes, traversed_edge_pixels = path_trace(skeleton_image)

    visualize_graph_on_image(G, skeleton_image, nodes, traversed_edge_pixels)
