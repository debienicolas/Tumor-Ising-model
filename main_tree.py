import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import pydot
from networkx.drawing.nx_agraph import graphviz_layout
import time 
from scipy.constants import Boltzmann
from tqdm import tqdm
import numba
from PIL import Image
import networkx as nx
import os
from datetime import datetime
import pandas as pd
from tree import kTree

class IsingModel:
    def __init__(self, nodes, neighbors, temperature=2.0, J=1.0, pos=None):
        """
        Initialize the Ising model for a 2D square lattice.
        Random spin initialisation.
        
        Args:
            size: Size of the square lattice
            temperature: Temperature parameter
            J: Coupling constant, positive for ferromagnetic coupling
        """
        self.temperature = temperature
        self.J = J
        self.beta = 1 / temperature # inverse temperature

        # Initialize random spin configuration
        self.spins = nodes
        self.neighbors = neighbors
        
        # For tracking observables
        self.energies = []
        self.magnetizations = []

        # for storing animation frames
        self.frames = []

        # For storing the final results
        self.spins_final = None
        self.energy_final = None
        self.magnetization_final = None

        # positions for graph plotting
        self.pos = pos


@numba.njit(nopython=True)
def calc_neighbor_sum(i:int, spins:np.ndarray, neighbors:np.ndarray) -> float:
    """
    Calculate the sum of the neighbors of node i.
    Works with a neighbors array where neighbors[i,:] contains the neighbors of node i.
    """
    node_neigh = neighbors[i,:]
    # don't consider the -1 as a neighbor
    result = np.sum(spins[node_neigh[node_neigh != -1]])
    return result

@numba.njit(nopython=True)
def calc_hamiltonian(spins:np.ndarray, neighbors:np.ndarray, J:float) -> float:
    """
    Calculate the Hamiltonian of the system.
    end 
    """
    H = 0
    for i in range(0,spins.shape[0]):
        neighs_sum = calc_neighbor_sum(i,spins, neighbors=neighbors)
        H += -J * spins[i] * neighs_sum
    return H / 2

@numba.njit(nopython=True)
def calc_magnetization(spins:np.ndarray) -> float:
    """
    Calculate the magnetization of the system.
    """
    return np.abs(np.sum(spins)/spins.size)

@numba.njit(nopython=True)
def calc_energy_diff(spins:np.ndarray, neighbors:np.ndarray, J:float, i:int) -> float:
    """
    Calculate the energy difference of flipping a spin.
    """
    neighbors_sum = calc_neighbor_sum(i,spins, neighbors=neighbors)
    return 2 * J * spins[i] * neighbors_sum

@numba.njit(nopython=True)
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

@numba.njit(nopython=True)
def simulate(spins:np.ndarray, neighbors:np.ndarray, J:float, beta:float, n_iterations:int) -> np.ndarray:
    """
    Simulate the Ising model.
    """
    # save 100 frames no matter the amount of iterations
    save_every = n_iterations // 100
    saved_frames = []

    for i in range(n_iterations):
        spins = metropolis_step(spins, neighbors, J, beta)
        if i % save_every == 0:
            saved_frames.append(spins.copy())
    return saved_frames

def simulate_ising_model(model: IsingModel, n_iterations: int=10_000) -> IsingModel:
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
    spins = model.spins
    neighbors = model.neighbors
    J = model.J
    beta = model.beta
    
    start_time = time.time()
    model.frames = simulate(spins, neighbors, J, beta, n_iterations)
    
    elapsed_time = time.time() - start_time
    print("Time taken: ", elapsed_time)

    model.spins_final = spins
    model.energy_final = calc_hamiltonian(spins, neighbors, J) / spins.size
    model.magnetization_final = calc_magnetization(spins)

    return model

def plot_graph(nodes:np.ndarray, neighbors:np.ndarray, ax=None, pos=None):
    # Use provided axes or create new figure
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
    
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
    # Draw nodes with colors based on spin values
    node_colors = ['blue' if spin == 1 else 'red' for spin in nodes]
    
    # Draw the network on the provided axes
    nx.draw(G, pos, ax=ax,
            node_color=node_colors,
            node_size=10,
            with_labels=False,
            font_color='white',
            font_weight='bold')
    
    # Add a legend
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                            markersize=10, label='Spin = +1')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                            markersize=10, label='Spin = -1')
    ax.legend(handles=[blue_patch, red_patch], loc='upper right')
    
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

def animate_ising_model(model: IsingModel, output_dir="output", T=None):
    """
    Animate the Ising model.
    Use the frames stored in model.frames.
    
    Args:
        model: IsingModel object
        output_dir: Directory where animation should be saved
    """
    # Create automatic filename
    tree_params = {}
    tree_params["iters"] = len(model.frames) * 100  # Multiply by frame save interval
    model.temperature = T
    plot_path = get_output_path(
        output_dir=output_dir, 
        model=model, 
        params=tree_params,
        prefix="animation", 
        extension=".gif"
    )
    
    print("Saving animation...")
    print(f"Number of frames: {len(model.frames)}")
    
    # Convert spin arrays to image format first
    image_frames = []
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # Process each frame
    for i, spin_array in enumerate(model.frames):
        ax.clear()  # Clear previous frame
        plot_graph(spin_array, model.neighbors, ax=ax, pos=model.pos)
        
        # Add frame number to title
        if T is not None:
            ax.set_title(f'Ising Model Simulation - T={T} - Frame {i} / {len(model.frames)-1}')
        else:
            ax.set_title(f'Ising Model Simulation - Frame {i} / {len(model.frames)-1}')
        
        # Render to image
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())
        image_frames.append(image)
    
    plt.close(fig)
    
    # Create GIF with PIL
    frames_pil = [Image.fromarray(frame) for frame in image_frames]
    frames_pil[0].save(
        plot_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=100,  # Time between frames in milliseconds (faster)
        loop=0
    )
    print(f"Animation saved as '{plot_path}'")

def simulate_ising_model_temps(temps: np.ndarray, nodes:np.ndarray, neighbors:np.ndarray, 
                              J: float=1.0, n_iterations: int=10_000, output_dir=None):
    """
    Simulate the Ising model for a range of temperatures.
    
    Args:
        temps: Array of temperatures to simulate
        nodes: Initial node configuration
        neighbors: Neighbor matrix
        J: Coupling constant
        n_iterations: Number of iterations per temperature
        output_dir: Directory where plots should be saved
    """
    models = []
    for temp in temps:
        model = IsingModel(nodes=nodes, neighbors=neighbors, temperature=temp, J=J)
        model = simulate_ising_model(model, n_iterations=n_iterations)
        print(f"Final model energy: {model.energy_final}, Final model magnetization: {model.magnetization_final}")
        models.append(model)
    
    # Create automatic filename
    params = {
        "temps": f"{temps.min():.1f}-{temps.max():.1f}",
        "iters": n_iterations,
        "J": J
    }
    if output_dir is not None:
        plot_path = get_output_path(
            output_dir=output_dir, 
            params=params,
            prefix="temp_sweep", 
            extension=".png"
        )
    
        # Plot the final energy and magnetization as a function of temperature
        f = plt.figure(figsize=(18, 10))
        sp = f.add_subplot(1, 2, 1)
        sp.scatter(temps, [model.energy_final for model in models], color="IndianRed", marker="o", s=50)
        sp.set_xlabel("Temperature")
        sp.set_ylabel("Energy")
        sp = f.add_subplot(1, 2, 2)
        sp.scatter(temps, [model.magnetization_final for model in models], color="RoyalBlue", marker="o", s=50)
        sp.set_xlabel("Temperature")
        sp.set_ylabel("Magnetization")
        plt.legend()
        plt.savefig(plot_path)
        
        print(f"Temperature sweep plot saved as '{plot_path}'")
        
        plt.show()
        plt.close()

    # return the numeric results in a dataframe
    results = pd.DataFrame({
        "temperature": temps,
        "energy": [model.energy_final for model in models],
        "magnetization": [model.magnetization_final for model in models],
    })
    return results

# Example usage
if __name__ == "__main__":
    pass
    # # Define output directory
    # output_dir = "variable_depth"
    
    #     # Temperature sweep
    # temps = np.linspace(0.1, 4.0, 100)
    # all_results = pd.DataFrame()
    # for depth in range(3, 6):
    #     tree = kTree(k=3, depth=depth)
    #     nodes, neighbors = tree.construct_tree()
    #     results = simulate_ising_model_temps(temps, nodes, neighbors, J=1.0, n_iterations=10_000)
    #     results["depth"] = depth
    #     results["k"] = 3
    #     all_results = pd.concat([all_results, results])
    # all_results.to_csv(f"{output_dir}/all_results.csv", index=False)

    # # plot the results in a 3x3 grid
    # f = plt.figure(figsize=(18, 10))
    # sp = f.add_subplot(1, 2, 1)
    # sp.scatter(all_results["temperature"], all_results["energy"], c=all_results["depth"], cmap="viridis", marker="o", s=50)
    # sp.set_xlabel("Temperature")
    # sp.set_ylabel("Energy")

    # sp = f.add_subplot(1, 2, 2)
    # sp.scatter(all_results["temperature"], all_results["magnetization"], c=all_results["depth"], cmap="viridis", marker="o", s=50)
    # sp.set_xlabel("Temperature")
    # sp.set_ylabel("Magnetization")

    # f.colorbar(sp.collections[0], ax=sp)

    # plt.show()
    # plt.close()
    

    
