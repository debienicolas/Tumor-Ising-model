import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from networkx.drawing.nx_agraph import graphviz_layout
import time 
from tqdm import tqdm
import numba
from PIL import Image
import networkx as nx
import os
from datetime import datetime
from branch_sim import MamSimulation
from collections import defaultdict
from numba import prange
from multiprocessing import Pool, cpu_count

from utils.gen_utils import graph_to_model_format
from utils.branch_sim_utils import convert_branch_coords_to_graph, model_format_at_time
from IsingModel import IsingModel 


from numba import njit



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

@numba.njit(nopython=True, fastmath=True)
def calc_hamiltonian(spins:np.ndarray, neighbors:np.ndarray, J:float, beta:float=None, h:float=0.0) -> float:
    """
    Calculate the Hamiltonian of the system.
    end 
    """
    H = 0
    # h is based on the beta. if beta is large h is 1 and goes to 0 when beta is 0.55 or smaller
    # if beta is not None:
    #     # Sigmoid-like transition: h goes from 0 to 1 as beta increases from 0.55
    #     # Using a smooth transition with steepness parameter
    #     h = 1.0 / (1.0 + np.exp(-10.0 * (beta - 0.34)))
    # else:
    #     h = 0.0
    
    for i in range(0,spins.shape[0]):
        node_neigh = neighbors[i,:]
        valid_neighs = node_neigh[node_neigh != -1]
        for j in valid_neighs:
            H += (-J * spins[i] * spins[j]) / 2
        # if it is a leaf node, add a small magnetic field to the energy
        if np.sum(neighbors[i,1:]) == (neighbors.shape[1]-1)*-1:
            H += -h * spins[i]
    return H 


@numba.njit(nopython=True, fastmath=True)
def calc_magnetization(spins:np.ndarray) -> float:
    """
    Calculate the magnetization of the system.
    """
    return np.abs(np.sum(spins))

@numba.njit(nopython=True, fastmath=True)
def calc_energy_diff(spins:np.ndarray, neighbors:np.ndarray, J:float, i:int, beta:float=None, h:float=0.0) -> float:
    """
    Calculate the energy difference of flipping a spin.
    """
    hamil_before = calc_hamiltonian(spins, neighbors, J, beta=beta, h=h)
    spins[i] *= -1
    hamil_after = calc_hamiltonian(spins, neighbors, J, beta=beta, h=h)
    return hamil_after - hamil_before

@numba.njit(nopython=True, fastmath=True)
def metropolis_step(spins: np.ndarray, neighbors: np.ndarray, J: float, beta: float, h:float) -> np.ndarray:
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
    for _ in range(spins.size):
        i = np.random.randint(0, spins.size)
        #print(i)
        energy_diff = calc_energy_diff(spins.copy(), neighbors, J, i, h=h)    
        # prob of flipping the spin = exp(-beta * delta_E) if delta_E > 0
        # flip if delta_E < 0 or exp(-beta * delta_E) > np.random.random()
        if energy_diff < 0 or np.exp(-beta * energy_diff) > np.random.random():
            spins[i] *= -1
    return spins

@numba.njit(nopython=True, fastmath=True)
def glauber_step(spins:np.ndarray, neighbors:np.ndarray, J:float, beta: float)-> np.ndarray:
    """
    Differs from the Metropolis step in that it uses the Fermi function to determine the probability of flipping the spin. 
    ref. https://en.wikipedia.org/wiki/Glauber_dynamics
    """
    for _ in range(spins.size):
        i = np.random.randint(0, spins.size)
        energy_diff = calc_energy_diff(spins, neighbors, J, i, beta=beta)
        # Differs from metropolist step: use the Fermi function
        prob = 1 / (1 + np.exp(energy_diff*beta))
        if np.random.random() < prob:
            spins[i] *= -1
    return spins

@numba.njit(nopython=True, fastmath=True, parallel=False)
def wolff_step(spins:np.ndarray, neighbors:np.ndarray, J:float, beta:float)-> np.ndarray:
    """
    Perform a single Wolff sweep on the system.
    """
    for _ in range(spins.size):
        seed_idx = np.random.randint(0, spins.size)
        # calculate the P_add
        P_add = 1 - np.exp(-2*J*beta)
        # Pick a random starting node
        seed_spin = spins[seed_idx]
        
        # Use arrays instead of checking sum each iteration
        cluster = np.zeros(spins.size, dtype=np.int8)
        frontier = np.zeros(spins.size, dtype=np.int8)
        
        # Add seed to cluster and frontier
        cluster[seed_idx] = 1
        frontier[seed_idx] = 1
        frontier_size = 1
        
        # Use a counter instead of np.sum each iteration
        while frontier_size > 0:
            # Find frontier nodes efficiently
            frontier_indices = np.where(frontier == 1)[0]
            # Pick the first one instead of a random choice (much faster)
            current = frontier_indices[0]
            frontier[current] = 0
            frontier_size -= 1
            
            # Get neighbors of current node
            node_neighs = neighbors[current]
            # Process only valid neighbors (-1 indicates no neighbor)
            valid_mask = node_neighs != -1
            valid_neighbors = node_neighs[valid_mask]
            
            # Find neighbors with same spin as seed that aren't already in cluster
            for neigh in valid_neighbors:
                if spins[neigh] == seed_spin and cluster[neigh] == 0:
                    # Add to cluster with probability P_add
                    if np.random.random() < P_add:
                        cluster[neigh] = 1
                        frontier[neigh] = 1
                        frontier_size += 1
        
        # Flip all spins in the cluster
        cluster_indices = np.where(cluster == 1)[0]
        spins[cluster_indices] *= -1
            
    return spins

#@numba.njit(nopython=True, fastmath=True, parallel=False)
def wolff_step_boundary(spins:np.ndarray, neighbors:np.ndarray, J:float, beta:float)-> np.ndarray:
    """
    Perform a single Wolff sweep on the system.
    Leaf nodes (nodes with only 1 valid neighbor) are not updated.
    """
    # First, identify leaf nodes
    leaf_nodes = np.zeros(spins.size, dtype=np.int8)
    for i in range(spins.size):
        valid_count = 0
        for j in range(neighbors.shape[1]):
            if neighbors[i, j] != -1:
                valid_count += 1
        if valid_count == 1:
            leaf_nodes[i] = 1
    
    # Get non-leaf nodes for seed selection
    non_leaf_indices = np.zeros(spins.size, dtype=np.int32)
    non_leaf_count = 0
    for i in range(spins.size):
        if leaf_nodes[i] == 0:
            non_leaf_indices[non_leaf_count] = i
            non_leaf_count += 1
    
    if non_leaf_count == 0:
        return spins  # All nodes are leaf nodes, nothing to update
    
    for _ in range(spins.size):
        # Pick seed only from non-leaf nodes
        seed_idx = non_leaf_indices[np.random.randint(0, non_leaf_count)]
        
        # calculate the P_add
        P_add = 1 - np.exp(-2*J*beta)
        # Pick a random starting node
        seed_spin = spins[seed_idx]
        
        # Use arrays instead of checking sum each iteration
        cluster = np.zeros(spins.size, dtype=np.int8)
        frontier = np.zeros(spins.size, dtype=np.int8)
        
        # Add seed to cluster and frontier
        cluster[seed_idx] = 1
        frontier[seed_idx] = 1
        frontier_size = 1
        
        # Use a counter instead of np.sum each iteration
        while frontier_size > 0:
            # Find frontier nodes efficiently
            frontier_indices = np.where(frontier == 1)[0]
            # Pick the first one instead of a random choice (much faster)
            current = frontier_indices[0]
            frontier[current] = 0
            frontier_size -= 1
            
            # Get neighbors of current node
            node_neighs = neighbors[current]
            # Process only valid neighbors (-1 indicates no neighbor)
            valid_mask = node_neighs != -1
            valid_neighbors = node_neighs[valid_mask]
            
            # Find neighbors with same spin as seed that aren't already in cluster
            for neigh in valid_neighbors:
                # Skip leaf nodes - they cannot be added to clusters
                if leaf_nodes[neigh] == 0 and spins[neigh] == seed_spin and cluster[neigh] == 0:
                    # Add to cluster with probability P_add
                    if np.random.random() < P_add:
                        cluster[neigh] = 1
                        frontier[neigh] = 1
                        frontier_size += 1
        
        # Flip all spins in the cluster (excluding leaf nodes)
        cluster_indices = np.where(cluster == 1)[0]
        for idx in cluster_indices:
            if leaf_nodes[idx] == 0:
                spins[idx] *= -1
            
    return spins

#@numba.njit(nopython=True, fastmath=True)
def simulate(spins:np.ndarray, neighbors:np.ndarray, J:float, beta:float, n_equilibration:int, n_mcmc:int, n_samples:int, n_sample_interval:int, step_algorithm:str, h:float=0.0) -> np.ndarray:
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
    assert neighbors.ndim == 2, "neighbors must be a 2D array"
    assert n_mcmc % n_sample_interval == 0, "n_mcmc must be divisible by n_sample_interval"
    # use just n_mcmc and sample_interval to calculate the number of samples
    n_samples = n_mcmc // n_sample_interval

    ##### Initialize the arrays for saving the results #####

    # equilibrium samples
    energy_equil = np.zeros(n_equilibration)
    magn_equil = np.zeros(n_equilibration)

    # mcmc samples
    spins_samples = np.zeros((n_samples, spins.size), dtype=np.int8)
    magn_samples = np.zeros(n_samples, dtype=np.float32)
    energy_samples = np.zeros(n_samples, dtype=np.float32)

    magn_all = np.zeros(n_mcmc)
    energy_all = np.zeros(n_mcmc)

    # used for calc. specific heat and susceptibilit -> using the MCMC samples
    E1,E2 = 0,0
    M1,M2 = 0,0
    

    ##### Equilibration #####
    for i in range(n_equilibration):
        if step_algorithm == "metropolis":
            spins = metropolis_step(spins, neighbors, J, beta,h)
        elif step_algorithm == "glauber":
            spins = glauber_step(spins, neighbors, J, beta)
        elif step_algorithm == "wolff":
            spins = wolff_step(spins, neighbors, J, beta)
        elif step_algorithm == "wolff_boundary":
            spins = wolff_step_boundary(spins, neighbors, J, beta)
            # check that the leaf nodes are +1
            for j in range(neighbors.shape[0]):
                if np.sum(neighbors[j,1:]) == (neighbors.shape[1]-1)*-1:
                    assert spins[j] == 1, "Leaf node is not +1"
                    
        # subselect the non_leaf spins to calculate the energy
        # mask = np.ones(spins.size, dtype=np.int8)
        # for j in range(neighbors.shape[0]):
        #     if np.sum(neighbors[j,1:]) == (neighbors.shape[1]-1)*-1:
        #         mask[j] = 0
        # non_leaf_spins = spins[mask == 1]
        # non_leaf_neighbors = neighbors[mask == 1]
        
        # save the equilibrium time steps
        energy_equil[i] = calc_hamiltonian(spins, neighbors, J, beta=beta, h=h) / spins.size
        magn_equil[i] = calc_magnetization(spins) / spins.size

    ##### MCMC #####
    for i in range(n_mcmc):
        if step_algorithm == "metropolis":
            spins = metropolis_step(spins, neighbors, J, beta, h=h)
        elif step_algorithm == "glauber":
            spins = glauber_step(spins, neighbors, J, beta)
        elif step_algorithm == "wolff":
            spins = wolff_step(spins, neighbors, J, beta)
        elif step_algorithm == "wolff_boundary":
            spins = wolff_step_boundary(spins, neighbors, J, beta)
            # check that the leaf nodes are +1 boundary conditions
            for j in range(neighbors.shape[0]):
                if np.sum(neighbors[j,1:]) == (neighbors.shape[1]-1)*-1:
                    assert spins[j] == 1, "Leaf node is not +1"
        
        # subselect the non_leaf spins to calculate the energy
        # mask = np.ones(spins.size, dtype=np.int8)
        # for j in range(neighbors.shape[0]):
        #     if np.sum(neighbors[j,1:]) == (neighbors.shape[1]-1)*-1:
        #         mask[j] = 0
        # non_leaf_spins = spins[mask == 1]
        # non_leaf_neighbors = neighbors[mask == 1]
        
        energy = calc_hamiltonian(spins, neighbors, J, beta=beta, h=h) / spins.size
        energy_all[i] = energy
        magn = calc_magnetization(spins)/spins.size
        magn_all[i] = magn
        
        # save the mcmc samples 
        if i % n_sample_interval == 0:
            sample_idx = i // n_sample_interval
            energy_samples[sample_idx] = energy
            magn_samples[sample_idx] = magn

            # update the used for calc. specific heat and susceptibilit -> using the MCMC samples
            E1 += energy
            E2 += energy**2
            M1 += magn
            M2 += magn**2

            # save the spins
            spins_samples[sample_idx] = spins.copy()
    

    ##### Calculate the results #####

    # average the magnetization and energy over the amount of samples
    avg_magn = np.mean(magn_samples)
    avg_energy = np.mean(energy_samples)

    if n_samples > 0:
        # calculate the specific heat
        specific_heat = ((E2/n_samples) - ((E1**2)/(n_samples**2)))*(beta**2) #)/spins.size)

        # calculate the susceptibility
        susceptibility = ((M2/n_samples) - ((M1**2)/(n_samples**2)))*(beta )#/ spins.size)

    else:
        specific_heat = None
        susceptibility = None

    return spins_samples, avg_magn, avg_energy, specific_heat, susceptibility, energy_samples, magn_samples, energy_equil, magn_equil, magn_all, energy_all



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
        spins_samples, avg_magn, avg_energy, specific_heat, susceptibility, energy_samples, magn_samples, energy_equil, magn_equil, magn_all, energy_all = simulate(
            spins = model.nodes, 
            neighbors = model.neighbors, 
            J = model.J, 
            beta = model.beta, 
            n_equilibration = model.n_equilib_steps, 
            n_mcmc = model.n_mcmc_steps,
            n_samples = model.n_samples,
            n_sample_interval = model.n_sample_interval,
            step_algorithm = model.step_algorithm,
            h = model.h)
        
    
    elapsed_time = time.time() - start_time
    print(f"Temperature: {model.temp} - Time taken: {elapsed_time}")
    

    #model.save_results(spins_collection, magn, energy, specific_heat, susceptibility)
    model.sample_spins = spins_samples
    model.avg_magn = avg_magn
    model.avg_energy = avg_energy
    model.specific_heat = specific_heat
    model.susceptibility = susceptibility
    model.energy_samples = energy_samples
    model.magn_samples = magn_samples
    model.energy_equil = energy_equil
    model.magn_equil = magn_equil
    model.magn_all = magn_all
    model.energy_all = energy_all

    # magn and energy have been averaged over a certain amount of samples
    return model

def simulate_ising_full(nodes:np.ndarray, neighbors:np.ndarray, J:float, n_equilib_steps:int, n_mcmc_steps:int, n_samples:int, n_sample_interval:int, temps:np.ndarray, step_algorithm:str, h=0.0, n_cores:int=1) -> IsingModel:
    """
    Simulate the entire ising model on all provided temps.
    General steps:
    - calculate the total amount of themolization/equilibration steps
    - calculate the autocorrelation time -> do this for each temp specifically
    - run the simulation for each temp in separate processes (make sure to precompile the njit functions)
    - save and plot the results
    """
    try:
        # create a pool of processes
        n_processes = n_cores
        print(f"Using {n_processes} processes")
        pool = Pool(n_processes)

        intermediate_results = []
        results = {}
        for t in tqdm(temps):
            # create a model
            nodes = np.random.choice([-1, 1], size=nodes.size)
            
            # set the leaf nodes to +1 boundary conditions
            if step_algorithm == "wolff_boundary":  
                for i in range(neighbors.shape[0]):
                    if np.sum(neighbors[i,1:]) == (neighbors.shape[1]-1)*-1:
                        nodes[i] = 1
            
            model = IsingModel(nodes=nodes, neighbors=neighbors, temp=t, J=J, 
                             n_equilib_steps=n_equilib_steps, n_mcmc_steps=n_mcmc_steps, 
                             n_samples=n_samples, n_sample_interval=n_sample_interval,
                             step_algorithm=step_algorithm, h=h)
            
            # run the simulation in parallel, collect the results
            intermediate_results.append(pool.apply_async(simulate_ising_model, args=(model,)))
            #curr_result = curr_result.get()
            #curr_t = curr_result.temp
            #results[curr_t] = curr_result
            #results[t] = pool.apply_async(simulate_ising_model, args=(model,))
        
        # Wait for all processes to complete and collect results
        for res in intermediate_results:
            curr_result = res.get()
            curr_t = curr_result.temp
            results[curr_t] = curr_result
        #final_results = {t: result.get() for t, result in results.items()}

        # sort the results by temp
        results = {t: results[t] for t in sorted(results.keys())}
        
        return results
        
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        raise
    finally:
        # Always clean up the pool
        pool.close()
        pool.join()

    

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

