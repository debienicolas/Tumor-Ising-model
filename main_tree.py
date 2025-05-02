import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import pydot
from networkx.drawing.nx_agraph import graphviz_layout
import time 
from tqdm import tqdm
import numba
from PIL import Image
import networkx as nx
import os
from datetime import datetime
import pandas as pd
from branch_sim import MamSimulation
from scipy.spatial import KDTree
from collections import defaultdict
from numba import prange

from utils.gen_utils import graph_to_model_format
from utils.branch_sim_utils import convert_branch_coords_to_graph, model_format_at_time
from IsingModel import IsingModel 


@numba.njit(nopython=True, fastmath=True)
def calc_neighbor_sum(i:int, spins:np.ndarray, neighbors:np.ndarray) -> float:
    """
    Calculate the sum of the neighbors of node i.
    Works with a neighbors array where neighbors[i,:] contains the neighbors of node i.
    """
    node_neigh = neighbors[i,:]
    # don't consider the -1 as a neighbor
    result = np.sum(spins[node_neigh[node_neigh != -1]])
    return result

@numba.njit(nopython=True, fastmath=True, parallel=True)
def calc_hamiltonian(spins:np.ndarray, neighbors:np.ndarray, J:float) -> float:
    """
    Calculate the Hamiltonian of the system.
    end 
    """
    H = 0
    for i in range(0,spins.shape[0]):
        neighs_sum = calc_neighbor_sum(i,spins, neighbors=neighbors)
        if np.isnan(neighs_sum):
            print(f"NaN neighbor sum at node {i}")
            continue
        H += -J * spins[i] * neighs_sum
    if H == 0:
        print("Zero Hamiltonian detected")
    return H / 2

@numba.njit(nopython=True, fastmath=True)
def calc_magnetization(spins:np.ndarray) -> float:
    """
    Calculate the magnetization of the system.
    """
    return np.abs(np.sum(spins)/spins.size)

@numba.njit(nopython=True, fastmath=True)
def calc_energy_diff(spins:np.ndarray, neighbors:np.ndarray, J:float, i:int) -> float:
    """
    Calculate the energy difference of flipping a spin.
    """
    neighbors_sum = calc_neighbor_sum(i,spins, neighbors=neighbors)
    return 2 * J * spins[i] * neighbors_sum

@numba.njit(nopython=True, fastmath=True)
def metropolis_step(spins: np.ndarray, neighbors: np.ndarray, J: float, beta: float) -> np.ndarray:
    """
    Perform a single Metropolis step on a randomly chosen spin.
    
    Args:
        spins: The spin configuration array
        J: Coupling constant
        beta: Inverse temperature (1/kT)
        rng_state: Random number generator state
        
    Returns:
        bool: True if the spin flip was accepted, False otherwise
    """
    for i in range(spins.size):
        energy_diff = calc_energy_diff(spins, neighbors, J, i)    
        # prob of flipping the spin = exp(-beta * delta_E) if delta_E > 0
        # flip if delta_E < 0 or exp(-beta * delta_E) > np.random.random()
        if energy_diff < 0 or np.exp(-beta * energy_diff) > np.random.random():
            spins[i] *= -1
    return spins

@numba.njit(nopython=True, fastmath=True)
def simulate(spins:np.ndarray, neighbors:np.ndarray, J:float, beta:float, n_equilibration:int, n_mcmc:int, n_samples:int, n_sample_interval:int) -> np.ndarray:
    """
    Simulate the Ising model.
    Save every spins during the mcmc steps.
    Return the final magnetization and energy over the amount of samples and interval.

    Returns:
        spins_collection: The collection of spins after each mcmc step
        mag: The magnetization of the system: average over the amount of samples
        energy: The energy of the system: average over the amount of samples
    """

    assert spins.ndim == 1, "spins must be a 1D array"

    # initialize the array to save all the mcmc spin configurations
    spins_collection = np.zeros((n_mcmc, spins.size), dtype=np.int8)

    # initialize the arrays to save the magnetization and energy
    sampl_magn = np.zeros(n_samples)
    sampl_energy = np.zeros(n_samples)

    # calc the start index for the samples starting from the last sample
    start_index = n_mcmc - n_samples * n_sample_interval
    
    for i in range(n_equilibration):
        spins = metropolis_step(spins, neighbors, J, beta)
    for i in range(n_mcmc):
        spins = metropolis_step(spins, neighbors, J, beta)
        if i % n_sample_interval == 0 and i >= start_index:
            sample_index = (i - start_index) // n_sample_interval
            sampl_magn[sample_index] = calc_magnetization(spins)
            sampl_energy[sample_index] = calc_hamiltonian(spins, neighbors, J) / spins.size
        # save the spins
        spins_collection[i] = spins.copy()
    
    # average the magnetization and energy over the amount of samples
    mag = np.mean(sampl_magn)
    energy = np.mean(sampl_energy)

    return spins_collection, mag, energy



def branch_evolve(spins:np.ndarray, coords:np.ndarray, evolve:np.ndarray, time_step:int, dist_thres:float=1.5, dim_cross:int=1) -> np.ndarray:
    """
    Evolve the branch by adding more nodes to the tips
    This is not the most efficient way to do this, but is good for now
    Oveview of the method:
    - construct the graph of the grown branch
    - update the corresponding nodes to with the existing spins values

    Args:
        spins: the spins of the current branch
        coords: the coordinates of the current branch
        evolve: the amount of nodes added at each time step
        time_step: the time step to evolve the branch to
        distance_threshold: the distance threshold for the graph
    """

    next_coords = coords[:np.sum(evolve[:time_step])]
    
    nodes, neighbors = model_format_at_time(next_coords, evolve, time_step, dim_cross=dim_cross)

    if dim_cross == 1:
        # update the spins of the current nodes -> spins maintained for the current branch
        nodes[:len(spins)] = spins
    else:
        # go over each layer and update the spins of the current nodes
        for i in range(dim_cross):
            n_nodes_per_layer_new = nodes.shape[0] // dim_cross
            n_nodes_per_layer_old = spins.shape[0] // dim_cross
            start_idx_new = i * n_nodes_per_layer_new
            start_idx_old = i * n_nodes_per_layer_old
            end_idx_old = start_idx_old + n_nodes_per_layer_old
            end_idx_new = start_idx_new + n_nodes_per_layer_old
            #print(start_idx_new, end_idx_new, start_idx_old, end_idx_old)
            
            nodes[start_idx_new:end_idx_new] = spins[start_idx_old:end_idx_old]
    return nodes, neighbors

    # # get the new coordinates of the branch
    # new_time = time_step
    # next_coords = get_branch_at_time(coords, evolve, new_time)
    # current_coords = get_branch_at_time(coords, evolve, time_step-1)
    # to_keep_coords = next_coords[current_coords.shape[0]:]


    # G = model.G
    # on_curr_nodes = len(G.nodes)
    # # set the spins of the current nodes
    # # create a tree of the next coordinates (all coordinates of the grown branch)
    # tree = KDTree(next_coords)
    # # add the new nodes to the graph
    # for i in range(len(to_keep_coords)):
    #     G.add_node(on_curr_nodes + i, pos=to_keep_coords[i])
    # # add the edges to the graph
    # for i in range(len(G.nodes)):
    #     # check if there are nodes within the distance threshold
    #     dists = tree.query_ball_point(G.nodes[i]["pos"], r=distance_threshold)
    #     for j in dists:
    #         if i != j and j not in G.neighbors(i):
    #             G.add_edge(i, j)
    
    # # now we have the graph of the grown branch
    # # conver the graph to the model format
    # new_spins = np.zeros(next_coords.shape[0])
    # new_spins[:len(current_coords)] = spins
    # new_spins[len(current_coords):] = np.random.choice([-1, 1], size=len(to_keep_coords))
    # _, new_neighbors = graph_to_model_format(G)

    # # update the model with the new graph and the new timestep
    # model.G = G
    # model.current_branch_time = new_time

    # assert len(new_spins) == new_neighbors.shape[0], f"Mismatch between spins and neighbors arrays: {len(new_spins)} != {new_neighbors.shape[0]}"
    # assert np.all(new_neighbors[new_neighbors != -1] < len(new_spins)), f"Invalid neighbor indices: {new_neighbors[new_neighbors != -1]}"
    # return new_spins, new_neighbors

def simulate_growing_ising_model(spins:np.ndarray, neighbors:np.ndarray, coords:np.ndarray, evolve:np.ndarray, J:float, beta:float,
                                 model:IsingModel) -> np.ndarray:
    """
    Simulate the Ising model with a growing branch.
    """
    
    start_time = model.start_time
    end_time = evolve.shape[0] - 1

    results = {}
    magn_results = defaultdict(list)
    energy_results = defaultdict(list)

    #G = convert_branch_coords_to_graph(coords, dist_thres=1.25)

    for t in range(start_time, end_time+1):
        print(f"Simulating branch at time {t}")
        results[t] = []
        
        spins_collection, magn, energy = simulate(spins, neighbors, J, beta, model.n_equilib_steps, model.n_mcmc_steps, model.n_samples, model.n_sample_interval)
            
        # stack the spins
        results[t] = spins_collection
        magn_results[t] = magn
        energy_results[t] = energy

        original_spins = spins.copy()
        # evolve the branch: take into account the case where stacked graph is used 
        spins, neighbors = branch_evolve(spins, coords, evolve, t+1, dim_cross=model.dim_cross)

        # assert that the first len(original_spins) spins are the same as the original spins
        if model.dim_cross == 1:
            assert np.all(spins[:len(original_spins)] == original_spins), "The first len(original_spins) spins are not the same as the original spins"
        

    return results, magn_results, energy_results

def simulate_ising_model(model: IsingModel) -> IsingModel:
    """
    Perform the Metropolis algorithm for the Ising model.
    Includes visualization of spin configurations during the simulation.
    
    Args:
        model: IsingModel object
        n_iterations: Number of Monte Carlo steps
        plot: If not None, save animation with this string in filename
        
    Returns:
        IsingModel: Final IsingModel object
    """
        
    start_time = time.time()
    # simulate the growing branch if the branch_sim is not None
    if model.coords is not None:
        spins_collection, magn, energy = simulate_growing_ising_model(model.nodes, model.neighbors, model.coords, model.evolve, model.J, model.beta, model)
        # in this case, the spins_collection is a dictionary with time as keys and spins 2d array as values
        # magn and energy are 
    else:
        spins_collection, magn, energy = simulate(
            spins = model.nodes, 
            neighbors = model.neighbors, 
            J = model.J, 
            beta = model.beta, 
            n_equilibration = model.n_equilib_steps, 
            n_mcmc = model.n_mcmc_steps,
            n_samples = model.n_samples,
            n_sample_interval = model.n_sample_interval)
        
    
    elapsed_time = time.time() - start_time
    print("Time taken: ", elapsed_time)
    

    model.save_results(spins_collection, magn, energy)

    # magn and energy have been averaged over a certain amount of samples
    return model

def plot_graph(nodes:np.ndarray, neighbors:np.ndarray, ax=None, G=None, draw_edges=True):
    """
    Plot the graph of the Ising model. 
    If the G is provided, use it to plot the graph with node colors being the spin values.
    If the graph is stacked, make sure to have done the offsetting of the positions
    If the G is not provided, create a new graph from the nodes and neighbors.
    """
    # Use provided axes or create new figure
    if ax is None:
        # check if the graph has a layers attribute -> stacked graph
        if "layers" in G.graph:
            fig = plt.figure(figsize=(15, 3*G.graph["layers"]))
            ax = fig.add_subplot(111)
        else:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
    if G is None:
        G = nx.Graph()  # Use undirected graph to avoid cycles
        # Add nodes with spin values as attributes
        for i, spin in enumerate(nodes):
            G.add_node(i, spin=int(spin))
        
        # Add edges from the neighbors array (avoiding duplicates)
        edges_added = set()
        for i in range(len(nodes)):
            for j in range(neighbors.shape[1]):
                neighbor = neighbors[i, j]
                if neighbor != -1:
                    # Only add edge if we haven't seen it before
                    edge = tuple(sorted([i, neighbor]))
                    if edge not in edges_added:
                        G.add_edge(i, neighbor)
                        edges_added.add(edge)
        if pos is None:
            pos = graphviz_layout(G, prog="twopi") # circo or twopi
    
    else:
        pos = nx.get_node_attributes(G, 'pos')
    # Draw nodes with colors based on spin values
    node_colors = ['blue' if spin == 1 else 'red' for spin in nodes]
    
    # Draw the network on the provided axes
    # if the graph is stacked, don't draw the edges, only the nodes
    if not draw_edges or "layers" in G.graph:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=10)
    else:
        nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=10, with_labels=False)
    
    # Add a legend
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                            markersize=10, label='Spin = +1')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                            markersize=10, label='Spin = -1')
    #ax.legend(handles=[blue_patch, red_patch])
    
    return ax

def get_output_path(output_dir, model=None, params=None, prefix="ising", extension=".png"):
    """
    Generate a standardized output path for experiment files.
    
    Args:
        output_dir: Directory where outputs should be saved
        model: Optional IsingModel object to extract parameters from
        params: Dictionary of additional parameters to include in filename
        prefix: String prefix for the filename
        extension: File extension (including the dot)
    
    Returns:
        String path to the output file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Start with the base components
    components = [prefix]
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    components.append(timestamp)
    
    # Add model parameters if provided
    if model is not None:
        components.append(f"T{model.temperature}")
        components.append(f"J{model.J}")
    
    # Add any additional parameters
    if params is not None:
        for key, value in params.items():
            components.append(f"{key}{value}")
    
    # Combine components into filename
    filename = "_".join(components) + extension
    return os.path.join(output_dir, filename)

def animate_ising_model(model: IsingModel, output_dir="output", animation_name=None, save_percent=10):
    """
    Animate the Ising model.
    Use the frames stored in model.frames.
    
    Args:
        model: IsingModel object
        output_dir: Directory where animation should be saved
    """
    # Create automatic filename
    if animation_name is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"ising_animation_{current_time}.gif")
    else:
        plot_path = os.path.join(output_dir, f"{animation_name}.gif")
    
    print("Saving animation...")

    ### use 10% of all the frames in the animation
    total_frames = model.spins.shape[0]
    n_frames = int(total_frames * (save_percent / 100))
    step = total_frames // n_frames
    frames = model.spins[::step]
    print(f"Number of frames: {len(frames)}")
    
    # Convert spin arrays to image format first
    image_frames = []
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    
    # Process each frame
    for i in tqdm(range(frames.shape[0])):
        spin_array = frames[i]
        ax.clear()  # Clear previous frame
        if model.coords is not None:
            plot_graph(spin_array, model.neighbors, ax=ax, G=model.G)
        else:
            plot_graph(spin_array, model.neighbors, ax=ax, G=model.G)
        
        # Add frame number to title
        ax.set_title(f'Ising Model Simulation - T={model.temp} - Frame {i} / {len(frames)}')
        
        # Render to image
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())
        image_frames.append(Image.fromarray(image))
    
    plt.close(fig)
    
    # Create GIF with PIL
    image_frames[0].save(
        plot_path,
        save_all=True,
        append_images=image_frames[1:],
        duration=100,  # Time between frames in milliseconds (faster)
        loop=0
    )
    print(f"Animation saved as '{plot_path}'")
    plt.close()

