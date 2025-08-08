import cv2
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from math import sqrt
from intersections import find_intersections


def find_neighbours(image, coordinates, traversed):
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


def dist(coord1, coord2):
    return sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def trace_one_path_bfs(start_pixel, source, image, nodes_list, proximity_threshold=5):
    """
    Performs a self-contained BFS to trace a single path from a starting
    pixel until it hits any node in nodes_set.
    """
    q = deque([(start_pixel, [start_pixel])])  # Queue: (current_pixel, path_taken)
    visited_in_this_path = {source, start_pixel}

    while q:
        current_pixel, path = q.popleft()

        for neighbor in find_neighbours(image, current_pixel, visited_in_this_path):
            for node_id, node_coord in enumerate(nodes_list):
                if node_coord == source:
                    continue
                if dist(neighbor, node_coord) < proximity_threshold:
                    # Found a junction! Return its ID and the path.
                    return node_id, path + [neighbor]

            visited_in_this_path.add(neighbor)
            q.append((neighbor, path + [neighbor]))

    return None, []  # Path did not lead to a junction


def path_trace_corrected(image):
    """
    Correctly builds the graph by tracing one full path at a time.
    """
    print("Finding intersections...")
    nodes = find_intersections(image)
    if not nodes:
        print("No nodes found.")
        return nx.Graph(), []

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
                        weight=dist(start_node_coord, nodes[end_node_id]),
                    )

                for pixel in path_pixels:
                    traversed_edge_pixels.add(pixel)

    print(G)
    visualize_graph_on_image(G, image, nodes)
    return G, nodes


def visualize_graph_on_image(graph, image, node_coordinates):
    """
    Draws a networkx graph overlaid on a background image.

    Args:
        graph (nx.Graph): The graph object to draw.
        image (np.ndarray): The background image (skeleton or original).
        node_coordinates (list): A list of (row, col) tuples. The index of
                                 each tuple must correspond to the node ID in the graph.
    """
    if not node_coordinates:
        print("Cannot visualize: No node coordinates provided.")
        return

    print("Generating visualization...")

    # Create the position dictionary needed by networkx.
    # The key is the node ID (0, 1, 2...), and the value is the (x, y) coordinate.
    # Remember: image (row, col) corresponds to plot (y, x).
    pos = {i: (coord[1], coord[0]) for i, coord in enumerate(node_coordinates)}

    # Create a new plot
    fig, ax = plt.subplots(figsize=(12, 12))

    # Display the background image
    ax.imshow(image, cmap="gray")

    # Draw the graph over the image
    nx.draw(
        graph,
        pos,
        ax=ax,
        with_labels=True,  # Show node IDs
        node_size=250,  # Make nodes bigger and easier to see
        node_color="cyan",  # A bright color for nodes
        edge_color="magenta",  # A bright color for edges
        width=2.5,  # Make lines thicker
        font_size=10,  # Font size for node labels
        font_color="black",  # A contrasting color for the labels
    )

    ax.set_title("Generated Graph on Skeleton Image", fontsize=16)
    plt.show()


def main():
    image_path = "/home/neutral/Documents/HR-0030-055-100478-R.dw_processed.png"

    # Load the image in grayscale
    try:
        skeleton_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if skeleton_image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        exit()

    path_trace_corrected(skeleton_image)


main()
