import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import time 
from scipy.constants import Boltzmann
from tqdm import tqdm
import numba
from PIL import Image
import pandas as pd
import os
# Create undecorated versions of all functions
def set_seed_impl(seed:int):
    """
    Set the seed for the random number generator.
    This only works with single threaded code -> see issue: https://github.com/numba/numba/issues/6002
    """
    np.random.seed(seed)

def calc_hamiltonian_impl(spins:np.ndarray, J:float) -> float:
    """
    Calculate the Hamiltonian of the system.
    Toroidal boundary conditions.
    Go over the boundary conditions seperately to avoid having to use modulo for every single position.
    """
    H = 0
    for i in range(0,spins.shape[0]):
        for j in range(0,spins.shape[1]):
            neighs_sum = (
                spins[i, (j+1) % spins.shape[1]] +
                spins[i, (j-1) % spins.shape[1]] +
                spins[(i+1) % spins.shape[0], j] +
                spins[(i-1) % spins.shape[0], j]
            )
            H += -J * spins[i,j] * neighs_sum
    
    return H / 2

def calc_magnetization_impl(spins:np.ndarray) -> float:
    """
    Calculate the magnetization of the system.
    """
    return np.abs(np.sum(spins)/spins.size)

def setup_neighbor_indices_impl(n,m):
    """
    Precompute the neigbor indices for a given grid size
    """
    up = np.zeros((n), dtype=np.int32)
    down = np.zeros((n), dtype=np.int32)
    left = np.zeros((m), dtype=np.int32)
    right = np.zeros((m), dtype=np.int32)

    # Setup indices with boundary conditions
    for i in range(n):
        up[i] = i - 1 if i > 0 else n - 1
        down[i] = i + 1 if i < n - 1 else 0
    
    for j in range(m):
        left[j] = j - 1 if j > 0 else m - 1
        right[j] = j + 1 if j < m - 1 else 0
        
    return up, down, left, right

def calc_energy_diff_impl(spins:np.ndarray, J:float, i:int, j:int, up:np.ndarray, down:np.ndarray, left:np.ndarray, right:np.ndarray) -> float:
    """
    Calculate the energy difference of flipping a spin.
    """
    n,m = spins.shape
    neighbors_sum = (
        spins[up[i], j] +
        spins[down[i], j] +
        spins[i, left[j]] +
        spins[i, right[j]]
    )
    return 2 * J * spins[i,j] * neighbors_sum

def metropolis_step_impl(spins: np.ndarray, J: float, beta: float, up: np.ndarray, down: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Perform a single Metropolis step on a randomly chosen spin.
    """
    n, m = spins.shape
    for i in range(n):
        for j in range(m):
            energy_diff = calc_energy_diff(spins, J, i, j, up, down, left, right)    
            if energy_diff < 0 or np.exp(-beta * energy_diff) > np.random.random():
                spins[i,j] *= -1
    return spins

# Create JIT-compiled versions
set_seed_numba = numba.njit(nopython=True)(set_seed_impl)
calc_hamiltonian_numba = numba.njit(nopython=True)(calc_hamiltonian_impl)
calc_magnetization_numba = numba.njit(nopython=True)(calc_magnetization_impl)
setup_neighbor_indices_numba = numba.njit(nopython=True)(setup_neighbor_indices_impl)
calc_energy_diff_numba = numba.njit(nopython=True)(calc_energy_diff_impl)
metropolis_step_numba = numba.njit(nopython=True)(metropolis_step_impl)

# Function pointers that can be switched at runtime
set_seed = set_seed_numba
calc_hamiltonian = calc_hamiltonian_numba
calc_magnetization = calc_magnetization_numba
setup_neighbor_indices = setup_neighbor_indices_numba
calc_energy_diff = calc_energy_diff_numba
metropolis_step = metropolis_step_numba

def use_numba(enabled=True):
    """Switch between Numba and non-Numba implementations"""
    global set_seed, calc_hamiltonian, calc_magnetization, setup_neighbor_indices, calc_energy_diff, metropolis_step
    
    if enabled:
        set_seed = set_seed_numba
        calc_hamiltonian = calc_hamiltonian_numba
        calc_magnetization = calc_magnetization_numba
        setup_neighbor_indices = setup_neighbor_indices_numba
        calc_energy_diff = calc_energy_diff_numba
        metropolis_step = metropolis_step_numba
        print("Using Numba acceleration")
    else:
        set_seed = set_seed_impl
        calc_hamiltonian = calc_hamiltonian_impl
        calc_magnetization = calc_magnetization_impl
        setup_neighbor_indices = setup_neighbor_indices_impl
        calc_energy_diff = calc_energy_diff_impl
        metropolis_step = metropolis_step_impl
        print("Running without Numba acceleration")

class IsingModel:
    def __init__(self, size=50, temperature=2.0, J=1.0):
        """
        Initialize the Ising model for a 2D square lattice.
        Random spin initialisation.
        
        Args:
            size: Size of the square lattice
            temperature: Temperature parameter
            J: Coupling constant, positive for ferromagnetic coupling
        """
        self.size = size
        self.temperature = temperature
        self.J = J

        # inverse temperature
        self.beta = 1 / temperature
        
        # Initialize random spin configuration
        self.spins = np.random.choice([-1, 1], size=(size, size))
        
        
        # For tracking observables
        self.energies = []
        self.magnetizations = []

        # for storing animation frames
        self.frames = []

        # For storing the final results
        self.spins_final = None
        self.energy_final = None
        self.magnetization_final = None

def animate_ising_model(model: IsingModel, plot: str=None):
    """
    Animate the Ising model.
    Use the frames stored in model.frames.
    """
    print("Saving animation...")
    print(f"Number of frames: {len(model.frames)}")
    
    # Convert spin arrays to image format first
    image_frames = []
    fig, ax = plt.subplots()
    
    # Add a colorbar once to ensure consistent size
    im = ax.imshow(model.frames[0], cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(im)
    fig.canvas.draw()  # Initial draw to get the figure size established
    
    # Process each frame
    for spin_array in model.frames:
        ax.clear()  # Clear previous frame
        im = ax.imshow(spin_array, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title('Ising Model Simulation')
        
        # Render to image
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_frames.append(frame)
    
    plt.close(fig)
    
    # Create GIF with PIL
    frames_pil = [Image.fromarray(frame) for frame in image_frames]
    frames_pil[0].save(
        f'ising_simulation_{plot}.gif',
        save_all=True,
        append_images=frames_pil[1:],
        duration=100,  # Time between frames in milliseconds (faster)
        loop=0
    )
    print(f"Animation saved as 'ising_simulation_{plot}.gif'")


def simulate_ising_model(model: IsingModel, n_iterations: int=10_000, plot: str=None) -> IsingModel:
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
    # save 5% of the iterations as frames
    save_every = 10

    spins = model.spins
    J = model.J
    beta = model.beta
    
    up, down, left, right = setup_neighbor_indices(spins.shape[0], spins.shape[1])
    start_time = time.time()
    # Main simulation loop
    for iter in tqdm(range(n_iterations)):
        # Perform single Metropolis step
        spins = metropolis_step(spins, J, beta, up, down, left, right)
            
        # Save frame every save_every iterations if plotting is enabled
        if plot and iter % save_every == 0:
            model.frames.append(spins.copy())
    
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    model.spins_final = spins
    model.energy_final = calc_hamiltonian(spins, J) / spins.size
    model.magnetization_final = calc_magnetization(spins)
            
    if plot:
        animate_ising_model(model, plot)
        # plot the final spin and save the plot
        plt.figure(figsize=(10, 10))
        plt.imshow(spins, cmap='RdBu', vmin=-1, vmax=1)
        plt.colorbar()
        plt.savefig(f'ising_simulation_{plot}_final.png')
        plt.close()
    return model

def simulate_ising_model_temps(temps: np.ndarray, grid_size: int=50, J: float=1.0, n_iterations: int=10_000) -> np.ndarray:
    """
    Simulate the Ising model for a range of temperatures.
    """
    models = []
    for temp in temps:
        model = IsingModel(size=grid_size, temperature=temp, J=J)
        model = simulate_ising_model(model, n_iterations=n_iterations)
        print(f"Final model energy: {model.energy_final}, Final model magnetization: {model.magnetization_final}")
        models.append(model)
    
    # plot the final energy and magnetization as a function of temperature
    # 2 plot side by side, scatter plots
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
    plt.savefig(f'ising_simulation_temps_{grid_size}x{grid_size}_{J}_{n_iterations}.png')
    


    # create a grid of some of the final spins for certain temperatures
    # take 10 equidistant temperatures from the temps array
    temps_to_plot = temps[::len(temps)//10]
    f2 = plt.figure(figsize=(20, 8))
    for i, temp in enumerate(temps_to_plot):
        model = next(model for model in models if model.temperature == temp)
        ax = f2.add_subplot(2, 5, i+1)
        im = ax.imshow(model.spins_final, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title(f'T = {temp:.2f}')
        ax.axis('off')

    # Add a colorbar that applies to all subplots
    cbar_ax = f2.add_axes([0.92, 0.15, 0.02, 0.7])
    f2.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
    plt.savefig(f'ising_simulation_temps_spins_{grid_size}x{grid_size}_{J}_{n_iterations}.png', dpi=150)
    
    plt.show()
    plt.close()
   

# Example usage
if __name__ == "__main__":
    # do a benchmark of using numba vs not using numba
    import time

    results = []

    # run using various grid sizes without numba
    use_numba(False)
    grid_sizes = [10, 50, 100, 200]
    for grid_size in grid_sizes:
        model = IsingModel(size=grid_size, temperature=0.1, J=1.0)
        start_time = time.time()
        model = simulate_ising_model(model, n_iterations=10_000)
        elapsed_time = time.time() - start_time
        results.append({"grid_size": grid_size, "elapsed_time": elapsed_time, "use_numba": False})
    
    # run using various grid sizes with numba
    use_numba(True)
    for grid_size in grid_sizes:
        model = IsingModel(size=grid_size, temperature=0.1, J=1.0)
        start_time = time.time()
        model = simulate_ising_model(model, n_iterations=10_000)
        elapsed_time = time.time() - start_time
        results.append({"grid_size": grid_size, "elapsed_time": elapsed_time, "use_numba": True})
    
    # plot the results
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 5))
    plt.scatter(df["grid_size"], df["elapsed_time"], c=df["use_numba"], cmap="viridis")
    plt.colorbar(label="Use Numba")
    plt.xlabel("Grid Size")
    plt.ylabel("Elapsed Time")
    plt.savefig("ising_benchmark.png")
    plt.show()

    # save the results to a csv file
    output_dir = os.path.join("bench_output", "ising_model")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "ising_model_benchmark.csv"), index=False)
