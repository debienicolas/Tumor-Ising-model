import networkx as nx
import numpy as np 
import pickle
import matplotlib.pyplot as plt
import random


def graph_to_model_format(G:nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the graph to the model format
    nodes: 1D array of node indices with spin up or down
    neighbors: 2D array of neighbor indices, take the amount of columns as the max number of neighbors, set the rest to -1
    """

    # if the node is a tuple, find the amount of layers 
    if isinstance(list(G.nodes)[0], tuple):
        # get the amount of layers
        n_layers = max(list(G.nodes), key=lambda x: x[1])[1] + 1
        nodes_per_layer = len(list(G.nodes)) // n_layers
        assert len(list(G.nodes)) % n_layers == 0, "The number of nodes must be divisible by the number of layers"

        # create the nodes array
        nodes = np.array([nodes_per_layer*y + x for x,y in list(G.nodes)])
        print("amount of layers", n_layers)
        print("nodes per layer", nodes_per_layer)

        spins = np.random.choice([-1, 1], size=len(nodes))
        
    else:
        nodes = np.array(list(G.nodes))
        spins = np.random.choice([-1, 1], size=len(nodes))
        
        # single graph case
        nodes_per_layer = nodes.size
        n_layers = 1
    print("nodes", nodes)
    
    # get the most connected node
    most_connected_node = max(G.nodes, key=lambda x: len(list(G.neighbors(x))))
    max_neighbors = len(list(G.neighbors(most_connected_node)))

    neighbors = np.full((len(nodes), max_neighbors), -1)
    for i, node in enumerate(nodes):
        if n_layers > 1:
            node = (node % nodes_per_layer, node // nodes_per_layer)
            neigh = list(G.neighbors(node))
            # transform the neighbors to the new node indices
            neigh = [nodes_per_layer*y + x for x,y in neigh]
            #print(i,node,neigh)
        else:
            neigh = list(G.neighbors(node))
        neighbors[i, :len(neigh)] = neigh
    
    return spins, neighbors


def dimensional_crossover(graph: nx.Graph, n_layers: int, pos_offset: int=30) -> nx.Graph:
    """
    Stack n_layers copies of the input graph, connecting corresponding nodes between layers.
    Each node is named as (original_node, layer) and has a 'level' attribute.
    """
    newGraph = nx.Graph()
    for layer in range(n_layers):
        for node, attrs in graph.nodes(data=True):
            new_node = (node, layer)
            # Copy node attributes and add 'level'
            new_attrs = attrs.copy()
            new_attrs['level'] = layer
            newGraph.add_node(new_node, **new_attrs)
        for u, v, attrs in graph.edges(data=True):
            newGraph.add_edge((u, layer), (v, layer), **attrs)
    # Connect corresponding nodes between layers
    for layer in range(n_layers - 1):
        for node in graph.nodes:
            newGraph.add_edge((node, layer), (node, layer + 1))
    
    # perform the pos offsetting
    pos = {}  
    x_offset = pos_offset/10
    y_offset = pos_offset
    for node, attrs in newGraph.nodes(data=True):
        orig_pos = attrs['pos']
        level = attrs['level']
        pos[node] = (orig_pos[0] + level*x_offset, orig_pos[1]+level*y_offset)
    nx.set_node_attributes(newGraph, pos, 'pos')


    # add the layers attribute to the graph
    newGraph.graph["layers"] = n_layers
    return newGraph


def spins_to_graph(spins: np.ndarray, graph: nx.Graph) -> nx.Graph:
    """
    Convert the spins and neighbors to a graph
    """
    pass


def plot_stacked_graph(graph, level_offset=20):
    """
    Plots a stacked graph where each layer is offset vertically by level_offset.
    Assumes each node is (node_id, level) and has a 'pos' attribute.
    """
    pos = {}
    for node, attrs in graph.nodes(data=True):
        orig_pos = attrs['pos']
        level = attrs['level']
        # Offset y by level*level_offset (or x if you prefer horizontal stacking)
        pos[node] = (orig_pos[0] + level * level_offset, orig_pos[1]+level*level_offset)
    
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(graph, pos=pos, node_size=10, with_labels=False)
    plt.show()


if __name__ == "__main__":
    # load the graph from the pickle file
    with open("output_mam_final_3/network_graph.pkl", "rb") as f:
        graph = pickle.load(f)

    # get a random node position
    node_pos = graph.nodes[np.random.randint(0, len(graph.nodes))]
    
    print("Len of graph", len(graph.nodes))

    # dimensional crossover
    new_graph = dimensional_crossover(graph, 3)

    #plot_stacked_graph(new_graph, level_offset=100)
    spins_simple,neighbors_simple = graph_to_model_format(graph)
    print("Original graph")
    print(spins_simple.shape)
    print(neighbors_simple.shape)

    spins, neighbors = graph_to_model_format(new_graph)
    print("Stacked graph")
    print(spins.shape)
    print(neighbors.shape)
