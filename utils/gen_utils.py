import networkx as nx
import numpy as np 
import pickle
import matplotlib.pyplot as plt
import random
import os


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
        spins = spins.astype(np.int8)
        
    else:
        nodes = np.array(list(G.nodes))
        spins = np.random.choice([-1, 1], size=len(nodes))
        spins = spins.astype(np.int8)
        
        # single graph case
        nodes_per_layer = nodes.size
        n_layers = 1
    
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


def dimensional_crossover(graph: nx.Graph, n_layers: int, pos_offset: int=100) -> nx.Graph:
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
    # add n_nodes_per_layer attribute to the graph
    newGraph.graph["n_nodes_per_layer"] = len(newGraph.nodes) // n_layers

    return newGraph


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


def plot_magn_energy(magn: np.ndarray|list, energy: np.ndarray|list, temps: np.ndarray|list, save_path:str=None, show: bool=True, title:str=None):
    """
    Plot the magnetization and energy as a function of temperature
    """
    plt.figure(figsize=(20, 10))
    # add a title to the figure if it is provided
    if title is not None:
        plt.suptitle(title)
    sp = plt.subplot(1, 2, 1)
    sp.scatter(temps, energy, label='energy', marker='o', color="IndianRed")
    sp.set_xlabel("Temperature")
    sp.set_ylabel("Energy")

    sp = plt.subplot(1, 2, 2)
    sp.scatter(temps, magn, label='magnetization', marker='o', color="RoyalBlue")
    sp.set_xlabel("Temperature")
    sp.set_ylabel("Magnetization")
    # set a scale for the y axis
    sp.set_ylim(0, 1)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    return

def plot_magn_energy_spech_susc(magn: np.ndarray|list, energy: np.ndarray|list, specific_heat: np.ndarray|list, susceptibility: np.ndarray|list, temps: np.ndarray|list, sizes: np.ndarray|list=None, save_path:str=None, show: bool=True, title:str=None, config:dict=None):
    """
    Plot the magnetization, energy, specific heat and susceptibility as a function of temperature
    
    """
    # convert the lists to numpy arrays
    magn = np.array(magn) if isinstance(magn, list) else magn
    energy = np.array(energy) if isinstance(energy, list) else energy
    specific_heat = np.array(specific_heat) if isinstance(specific_heat, list) else specific_heat
    susceptibility = np.array(susceptibility) if isinstance(susceptibility, list) else susceptibility
    temps = np.array(temps) if isinstance(temps, list) else temps

    # if multidimensional, the sizes should be given
    if magn.ndim > 1 or energy.ndim > 1 or specific_heat.ndim > 1 or susceptibility.ndim > 1:
        # all the arrays should have the same size
        assert magn.shape == energy.shape == specific_heat.shape == susceptibility.shape, "All the arrays should have the same size"
        assert sizes is not None, "If the data is multidimensional, the sizes should be given"
        assert len(sizes) == magn.shape[0], "The sizes should have the same length as the number of temperatures"
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    
    # plot the energy
    # Handle both 1D and multidimensional data
    if magn.ndim == 1:
        # Single curve case
        axs[0,0].plot(temps, energy, label='Energy')
        axs[0,1].plot(temps, magn, label='Magnetization')
        axs[1,0].plot(temps, specific_heat, label='Specific Heat')
        axs[1,1].plot(temps, susceptibility, label='Susceptibility')
    else:
        # Multiple curves case (one for each size)
        for i, size in enumerate(sizes):
            axs[0,0].plot(temps, energy[i], label=f'Size={size}')
            axs[0,1].plot(temps, magn[i], label=f'Size={size}')
            axs[1,0].plot(temps, specific_heat[i], label=f'Size={size}')
            axs[1,1].plot(temps, susceptibility[i], label=f'Size={size}')
    

    # Set labels and titles
    axs[0,0].set_ylabel('Energy')
    axs[0,1].set_ylabel('Magnetization')
    axs[1,0].set_xlabel('Temperature')
    axs[1,0].set_ylabel('Specific Heat')
    axs[1,1].set_xlabel('Temperature')
    axs[1,1].set_ylabel('Susceptibility')

    # Set overall title
    if title is not None:
        plt.suptitle(title)
    else:
        plt.suptitle('Ising Model Properties')

    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path)
    
    # Show plot if requested
    if show:
        plt.show()
    
    return 




def create_lattice(size:int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a lattice of size size x size
    """
    # nodes is a 1D array of rows appended to each other
    nodes = np.ones(size*size)
    # neighbors is a 2D array with the indices of the neighbors
    # in a grid each node has 4 neighbors
    neighbors = np.zeros((size*size,4))
    # loop over the nodes and assign the neighbors, use warping to handle the edges
    for i in range(size*size):
        col_index = i % size
        row_index = i // size

        neighbor_up = (i-size) if i-size >= 0 else (size*size - size + i)
        neighbor_down = (i+size) if i+size < size*size else i+size-size*size
        neighbor_left = (i-1) if col_index != 0 else size*(row_index+1)-1
        neighbor_right = (i+1) if col_index != size-1 else size*(row_index)
        neighbors[i] = np.array([neighbor_up, neighbor_down, neighbor_left, neighbor_right])
    # turn neighbors into list of ints
    neighbors = neighbors.astype(np.int32)
    return nodes, neighbors

def create_3D_lattice(size:int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a 3D lattice of size size x size x size
    """
    # nodes is a 1D array representing all nodes in the 3D lattice
    nodes = np.ones(size*size*size)
    # neighbors is a 2D array with the indices of the neighbors
    # in a 3D grid each node has 6 neighbors (up, down, left, right, forward, backward)
    neighbors = np.zeros((size*size*size, 6))
    
    # loop over the nodes and assign the neighbors, use wrapping to handle the edges
    for i in range(size*size*size):
        # Convert 1D index to 3D coordinates
        x = i % size
        y = (i // size) % size
        z = i // (size * size)
        
        # Calculate neighbors with periodic boundary conditions
        # Left and right (x direction)
        neighbor_left = z * size * size + y * size + ((x - 1) % size)
        neighbor_right = z * size * size + y * size + ((x + 1) % size)
        
        # Up and down (y direction) 
        neighbor_down = z * size * size + ((y - 1) % size) * size + x
        neighbor_up = z * size * size + ((y + 1) % size) * size + x
        
        # Forward and backward (z direction)
        neighbor_backward = ((z - 1) % size) * size * size + y * size + x
        neighbor_forward = ((z + 1) % size) * size * size + y * size + x
        
        neighbors[i] = np.array([neighbor_up, neighbor_down, neighbor_left, neighbor_right, neighbor_forward, neighbor_backward])
    
    # turn neighbors into list of ints
    neighbors = neighbors.astype(np.int32)
    return nodes, neighbors


def create_binary_tree(depth:int):
    # Create a binary tree using NetworkX
    G = nx.balanced_tree(2, depth-1)  # 2 children per node, depth-1 levels
    
    # Initialize nodes array (all nodes start with value 1)
    nodes = np.ones(len(G))
    
    # Initialize neighbors array
    neighbors = np.zeros((len(G), 3))  # Changed to 3 to store [parent, child1, child2]
    neighbors.fill(-1)  # Default to no neighbors
    
    # For each node, get its neighbors (both children and parent)
    for node in sorted(G.nodes()):
        # Get all neighbors (both children and parent)
        all_neighbors = sorted(list(G.neighbors(node)))
        
        # For root node (node 0), it only has children
        if node == 0:
            if len(all_neighbors) == 2:
                neighbors[node] = np.array([-1, all_neighbors[0], all_neighbors[1]])
        # For other nodes, they have one parent and potentially two children
        else:
            # Find the parent (it's the neighbor with smaller index)
            parent = min(all_neighbors)
            # Find the children (they're the neighbors with larger indices)
            children = [n for n in all_neighbors if n > node]
            
            # Store parent and children
            if len(children) == 2:
                neighbors[node] = np.array([parent, children[0], children[1]])
            elif len(children) == 1:
                neighbors[node] = np.array([parent, children[0], -1])
            else:
                neighbors[node] = np.array([parent, -1, -1])
    
    # turn neighbors into list of ints
    neighbors = neighbors.astype(np.int32)
    
    # Debug print
    # print("Tree structure:")
    # for node in sorted(G.nodes()):
    #     print(f"Node {node}: [parent, child1, child2] = {neighbors[node]}")
    
    return nodes, neighbors

def create_d_ary_tree(depth: int, d: int = 2):
    """
    Create a d-ary regular tree of given depth.
    
    Parameters:
    -----------
    depth : int
        The depth of the tree (number of levels)
    d : int
        The number of children per node (degree of the tree)
        
    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        nodes: 1D array of node values (all initialized to 1)
        neighbors: 2D array where each row contains [parent, child1, child2, ..., child_d]
                  with -1 for non-existent neighbors
    """
    # Create a d-ary tree using NetworkX
    G = nx.balanced_tree(d, depth-1)  # d children per node, depth-1 levels
    
    # Initialize nodes array (all nodes start with value 1)
    nodes = np.ones(len(G))
    
    # Initialize neighbors array: 1 parent + d children = d+1 total
    neighbors = np.zeros((len(G), d+1))
    neighbors.fill(-1)  # Default to no neighbors
    
    # For each node, get its neighbors (both children and parent)
    for node in sorted(G.nodes()):
        # Get all neighbors (both children and parent)
        all_neighbors = sorted(list(G.neighbors(node)))
        
        # For root node (node 0), it only has children
        if node == 0:
            # Store children in positions 1 to d+1
            neighbor_row = [-1]  # No parent for root
            for i in range(d):
                if i < len(all_neighbors):
                    neighbor_row.append(all_neighbors[i])
                else:
                    neighbor_row.append(-1)
            neighbors[node] = np.array(neighbor_row)
            
        # For other nodes, they have one parent and potentially d children
        else:
            # Find the parent (it's the neighbor with smallest index)
            parent = min(all_neighbors)
            # Find the children (they're the neighbors with larger indices)
            children = [n for n in all_neighbors if n > node]
            
            # Store parent and children: [parent, child1, child2, ..., child_d]
            neighbor_row = [parent]
            for i in range(d):
                if i < len(children):
                    neighbor_row.append(children[i])
                else:
                    neighbor_row.append(-1)
            neighbors[node] = np.array(neighbor_row)
    
    # Convert neighbors to int32
    neighbors = neighbors.astype(np.int32)
    
    return nodes, neighbors


def autocorrelation(x, max_lag=None):
    """
    Calculate the normalized autocorrelation function as defined in the formula:
    C(t) = [1/(N-t) * Σ X_i X_{i+t} - ⟨X⟩²] / [⟨X²⟩ - ⟨X⟩²]
    
    returns lags and autocorrelation values
    """
    x = np.asarray(x)
    N = len(x)
    
    if max_lag is None:
        max_lag = N // 4  # A common practice to avoid poor statistics at large lags
    
    # Calculate the mean and variance
    mean_x = np.mean(x)
    var_x = np.mean(x**2) - mean_x**2
    
    # Initialize autocorrelation array
    corr = np.zeros(max_lag + 1)
    
    # Calculate autocorrelation for each lag
    for t in range(max_lag + 1):
        # Calculate the sum of products X_i * X_{i+t}
        sum_prod = np.sum(x[:N-t] * x[t:]) / (N - t)
        
        # Apply the formula
        corr[t] = (sum_prod - mean_x**2) / var_x
    
    return np.arange(max_lag + 1), corr


def autocorr(x):
    max_lag = x.size//4
    x = np.array(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]
    result /= result[0]
    return result[:max_lag]


