import cv2
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from math import sqrt
from intersections import find_intersections_via_hit_or_miss


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


def is_close_to_edge(coord, edges):
    for item in edges:
        if dist(coord, item) < 5:
            return item
    return 0


def path_trace(image):
    "Creates a graph based on the skeletonized wing image"
    edges = find_intersections_via_hit_or_miss(image)
    print(edges)
    edges_set = set(edges)
    G = nx.Graph()
    # Add all the intersection points as edges
    for edge in edges:
        G.add_node(edges.index(edge))

    start_pixel = edges[0]
    traversed = set()
    Q = deque()
    traversed.add(start_pixel)
    Q.append((start_pixel, 0))

    while Q:
        pixel_coord, source_node_id = Q.popleft()
        neighbours = find_neighbours(image, pixel_coord, traversed)

        for item in neighbours:
            traversed.add(item)

            if close_edge := is_close_to_edge(item, edges_set):
                end_node_id = edges.index(close_edge)
                if end_node_id != source_node_id:
                    G.add_edge(
                        end_node_id,
                        source_node_id,
                        weight=dist(item, edges[source_node_id]),
                    )
                Q.append((item, end_node_id))
            else:
                Q.append((item, source_node_id))

    print(G.edges)
    visualize_graph_on_image(G, image, edges)
    return G


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

    print(path_trace(skeleton_image))


main()
