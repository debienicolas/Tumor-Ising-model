import networkx as nx
import numpy as np


def graph_to_model_format(G:nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the graph to the model format
    nodes: 1D array of node indices with spin up or down
    neighbors: 2D array of neighbor indices, take the amount of columns as the max number of neighbors, set the rest to -1
    """
    nodes = np.array(list(G.nodes))
    spins = np.random.choice([-1, 1], size=len(nodes))
    # get the most connected node
    most_connected_node = max(G.nodes, key=lambda x: len(list(G.neighbors(x))))
    max_neighbors = len(list(G.neighbors(most_connected_node)))

    neighbors = np.full((len(nodes), max_neighbors), -1)
    for i, node in enumerate(nodes):
        neigh = list(G.neighbors(node))
        neighbors[i, :len(neigh)] = neigh
    
    return spins, neighbors

def spins_to_graph(spins: np.ndarray, graph: nx.Graph) -> nx.Graph:
    """
    Convert the spins and neighbors to a graph
    """
    pass