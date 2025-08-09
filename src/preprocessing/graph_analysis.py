import cv2
import networkx as nx
import numpy as np
import os
import sys
import warnings
from graph_gen import path_trace
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import calculate_area


def _calculate_moments(data_list):
    """Helper function to calculate the first four statistical moments."""
    if not data_list or len(data_list) == 0:
        return [np.nan] * 4  # Return NaN if the list is empty

    mean = np.mean(data_list)
    variance = np.var(data_list)
    skewness = stats.skew(data_list)
    kurt = stats.kurtosis(data_list)

    return [mean, variance, skewness, kurt]


def calculate_wing_features(G, wing_area):
    """
    Calculates a 45-dimensional feature vector for a bee wing graph.

    The graph's nodes represent vein intersections, and edges represent vein
    segments. Edges are expected to have a 'length' attribute representing
    the segment's length in pixels.

    Args:
        G (nx.Graph): A NetworkX graph representing the wing venation.
        wing_area (float): The total area of the wing in pixels, used for normalization.

    Returns:
        dict: A dictionary containing the 45 calculated features, clearly named.
              Returns None if the input graph is empty.
    """
    if not G or G.number_of_nodes() == 0:
        warnings.warn("Input graph is empty. Cannot calculate features.")
        return {}

    features = {}

    # --- CLASS 1: Vein Intersection Density (1 feature) ---
    # This quantifies the complexity of the vein network relative to wing size.
    num_intersections = G.number_of_nodes()
    features["density_intersection"] = (
        num_intersections / wing_area if wing_area > 0 else 0
    )

    # --- CLASS 2: Segment Length Statistics (4 features) ---
    # These features capture the geometric properties of the vein segments.
    try:
        segment_lengths = [
            data["length"] for _, _, data in G.edges(data=True) if "length" in data
        ]
        if not segment_lengths:
            raise KeyError("No edges have the 'length' attribute.")
    except KeyError as e:
        raise ValueError(
            f"Error accessing edge lengths: {e}. Please ensure edges have a 'length' attribute."
        )

    moments_lengths = _calculate_moments(segment_lengths)
    features["length_mean"] = moments_lengths[0]
    features["length_variance"] = moments_lengths[1]
    features["length_skewness"] = moments_lengths[2]
    features["length_kurtosis"] = moments_lengths[3]

    # --- CLASS 3: Graph-Theoretic Properties (40 features) ---

    # == Subcategory 3a: Centrality Moments (16 features) ==
    # These features measure the "importance" of nodes in the network from different perspectives.

    # Degree Centrality (connections per node)
    degree_centrality = list(nx.degree_centrality(G).values())
    moments_degree = _calculate_moments(degree_centrality)
    (
        features["degree_mean"],
        features["degree_variance"],
        features["degree_skewness"],
        features["degree_kurtosis"],
    ) = moments_degree

    # Betweenness Centrality (how often a node is on the shortest path between others)
    betweenness_centrality = list(
        nx.betweenness_centrality(G, weight="length").values()
    )
    moments_betweenness = _calculate_moments(betweenness_centrality)
    (
        features["betweenness_mean"],
        features["betweenness_variance"],
        features["betweenness_skewness"],
        features["betweenness_kurtosis"],
    ) = moments_betweenness

    # Closeness Centrality (how close a node is to all others)
    closeness_centrality = list(nx.closeness_centrality(G, distance="length").values())
    moments_closeness = _calculate_moments(closeness_centrality)
    (
        features["closeness_mean"],
        features["closeness_variance"],
        features["closeness_skewness"],
        features["closeness_kurtosis"],
    ) = moments_closeness

    # Eigenvector Centrality (influence of a node in the network)
    try:
        eigenvector_centrality = list(
            nx.eigenvector_centrality(G, weight="length", max_iter=1000).values()
        )
        moments_eigenvector = _calculate_moments(eigenvector_centrality)
        (
            features["eigenvector_mean"],
            features["eigenvector_variance"],
            features["eigenvector_skewness"],
            features["eigenvector_kurtosis"],
        ) = moments_eigenvector
    except nx.PowerIterationFailedConvergence:
        warnings.warn(
            "Eigenvector centrality did not converge. Filling features with NaN."
        )
        (
            features["eigenvector_mean"],
            features["eigenvector_variance"],
            features["eigenvector_skewness"],
            features["eigenvector_kurtosis"],
        ) = [np.nan] * 4

    # == Subcategory 3b: Global Network Metrics (24 features) ==

    # -- Moments from Distributions (8 features) --

    # Local Clustering Coefficient Distribution
    local_clustering_coeffs = list(nx.clustering(G, weight="length").values())  # type: ignore
    moments_clustering = _calculate_moments(local_clustering_coeffs)
    (
        features["local_clustering_mean"],
        features["local_clustering_variance"],
        features["local_clustering_skewness"],
        features["local_clustering_kurtosis"],
    ) = moments_clustering

    # All-Pairs Shortest Path Length Distribution
    # This can be computationally expensive.
    path_lengths = []
    # Check for connectivity before calculating path-dependent metrics
    is_connected = nx.is_connected(G)
    if is_connected:
        for source, targets in nx.all_pairs_dijkstra_path_length(G, weight="length"):
            for target, length in targets.items():
                if source != target:
                    path_lengths.append(length)

    moments_paths = _calculate_moments(
        path_lengths
    )  # Will return NaNs if not connected
    (
        features["path_length_mean"],
        features["path_length_variance"],
        features["path_length_skewness"],
        features["path_length_kurtosis"],
    ) = moments_paths

    # -- Collection of Single-Value Global Metrics (16 features) --
    features["global_node_count"] = G.number_of_nodes()
    features["global_edge_count"] = G.number_of_edges()
    features["global_graph_density"] = nx.density(G)

    # Some metrics are only defined for connected graphs.
    if is_connected:
        # Diameter and radius for weighted graphs must be calculated from path lengths
        eccentricity = nx.eccentricity(
            G, sp=dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))
        )
        features["global_diameter"] = nx.diameter(G, e=eccentricity)
        features["global_radius"] = nx.radius(G, e=eccentricity)
        features["global_wiener_index"] = nx.wiener_index(G, weight="length")
    else:
        features["global_diameter"] = np.nan
        features["global_radius"] = np.nan
        features["global_wiener_index"] = np.nan
        warnings.warn(
            "Graph is not connected. Diameter, Radius, and Wiener Index are undefined and set to NaN."
        )

    features["global_assortativity"] = nx.degree_assortativity_coefficient(
        G, weight="length"
    )
    features["global_transitivity"] = nx.transitivity(
        G
    )  # Global clustering coefficient
    features["global_avg_clustering"] = nx.average_clustering(G, weight="length")
    features["global_efficiency"] = nx.global_efficiency(G)
    features["global_num_connected_components"] = nx.number_connected_components(G)

    # Spectral properties from the Laplacian matrix
    try:
        features["global_algebraic_connectivity"] = nx.algebraic_connectivity(
            G, weight="length"
        )
    except nx.NetworkXError:
        warnings.warn(
            "Could not compute algebraic connectivity (perhaps graph is too small). Setting to NaN."
        )
        features["global_algebraic_connectivity"] = np.nan

    # Modularity requires finding communities first
    try:
        communities = nx.community.greedy_modularity_communities(G, weight="length")
        features["global_modularity"] = nx.community.modularity(
            G, communities, weight="length"
        )
    except Exception:  # Broad exception for various community detection issues
        warnings.warn("Could not compute modularity. Setting to NaN.")
        features["global_modularity"] = np.nan

    features["global_num_triangles"] = sum(nx.triangles(G).values()) // 3  # type: ignore

    features["global_avg_node_connectivity"] = nx.average_node_connectivity(G)
    # Spectral radius (largest eigenvalue of the adjacency matrix)
    # This can be computationally intensive and may produce complex numbers
    try:
        adj_spectrum = nx.adjacency_spectrum(G, weight="length")
        features["global_spectral_radius"] = float(np.max(np.abs(adj_spectrum)))
    except Exception:
        warnings.warn("Could not compute spectral radius. Setting to NaN.")
        features["global_spectral_radius"] = np.nan

    # Final check to ensure we have exactly 45 features
    if len(features) != 45:
        warnings.warn(
            f"Expected 45 features, but generated {len(features)}. Please check the implementation."
        )

    return features


if __name__ == "__main__":
    # --- DEMONSTRATION ---
    image_path = "/home/neutral/Documents/HR-0030-055-100478-R.dw_processed.png"

    # Load the image in grayscale
    try:
        skeleton_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if skeleton_image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        exit()

    G_sample, _, _ = path_trace(skeleton_image)

    # Define a mock wing area
    mock_wing_area = calculate_area(skeleton_image)

    print(
        f"Sample graph created with {G_sample.number_of_nodes()} nodes and {G_sample.number_of_edges()} edges."
    )
    print("-" * 30)

    # Calculate the 45-dimensional feature vector
    wing_feature_vector = calculate_wing_features(G_sample, mock_wing_area)

    # Print the results
    if wing_feature_vector:
        print(f"Successfully calculated {len(wing_feature_vector)} features.")
        for i, (key, value) in enumerate(wing_feature_vector.items()):
            print(f"{i + 1:2d}. {key:<35}: {value:.4f}")
    else:
        print("Feature calculation failed.")
