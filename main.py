import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time 
from tqdm import tqdm
import numba
import os
from collections import deque

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
        self.spins = np.ones((size, size))
        
        
        # For tracking observables
        self.energies = []
        self.magnetizations = []

        # for storing animation frames
        self.frames = []

        # For storing the final results
        self.spins_final = None
        self.energy_final = None
        self.magnetization_final = None

@numba.njit(nopython=True)
def set_seed(seed:int):
    """
    Set the seed for the random number generator.
    This only works with single threaded code -> see issue: https://github.com/numba/numba/issues/6002
    """
    np.random.seed(seed)

@numba.njit(nopython=True)
def calc_hamiltonian(spins:np.ndarray, J:float) -> float:
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

@numba.njit(nopython=True)
def calc_magnetization(spins:np.ndarray) -> float:
    """
    Calculate the magnetization of the system.
    """
    return np.abs(np.sum(spins))


@numba.njit(nopython=True)
def setup_neighbor_indices(n,m):
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

@numba.njit(nopython=True)
def calc_energy_diff(spins:np.ndarray, J:float, i:int, j:int, up:np.ndarray, down:np.ndarray, left:np.ndarray, right:np.ndarray) -> float:
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

@numba.njit(nopython=True)
def metropolis_step(spins: np.ndarray, J: float, beta: float, up: np.ndarray, down: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
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
    n, m = spins.shape
    for i in range(n):
        for j in range(m):
            i = np.random.randint(0, n)  # Using Numba-compatible random functions
            j = np.random.randint(0, m)

            energy_diff = calc_energy_diff(spins, J, i, j, up, down, left, right)    
            # prob of flipping the spin = exp(-beta * delta_E) if delta_E > 0
            # flip if delta_E < 0 or exp(-beta * delta_E) > np.random.random()
            if energy_diff < 0 or np.exp(-beta * energy_diff) > np.random.random():
                spins[i,j] *= -1
    return spins


def wolff_cluster_update(grid, temperature, J):
    """Perform one Wolff cluster update."""
    for _ in range(grid.shape[0]):
        for _ in range(grid.shape[1]):
            size = grid.shape[0]
            
            # Choose a random starting point
            i = np.random.randint(0, size)
            j = np.random.randint(0, size)
            
            # Initialize the cluster
            cluster = set()
            to_visit = deque([(i, j)])
            cluster.add((i, j))
            
            # Probability of adding a spin to the cluster
            p = 1.0 - np.exp(-2.0 * J / temperature)
            
            while to_visit:
                i, j = to_visit.popleft()
                spin = grid[i, j]
                
                # Check all four neighbors
                neighbors = [
                    (i, (j - 1) % size),  # left
                    (i, (j + 1) % size),  # right
                    ((i - 1) % size, j),  # up
                    ((i + 1) % size, j)   # down
                ]
                
                for ni, nj in neighbors:
                    if (ni, nj) not in cluster and grid[ni, nj] == spin:
                        if np.random.random() < p:
                            cluster.add((ni, nj))
                            to_visit.append((ni, nj))
            
            # Flip all spins in the cluster
            for i, j in cluster:
                grid[i, j] *= -1
    
    return grid

def animate_ising_model(model: IsingModel, T: float=None, plot: str=None):
    """
    Animate the Ising model.
    Use the frames stored in model.frames.
    """
    print("Saving animation...")
    print(f"Number of frames: {len(model.frames)}")
    
    if len(model.frames) <= 1:
        print("Warning: Only one frame found, cannot create animation")
        return
    
    # Create figure with proper layout for colorbar
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Add a colorbar that persists across frames
    # First create a dummy image to initialize the colorbar
    im = ax.imshow(model.frames[0], cmap='RdBu', vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    
    # Function to update the plot for each frame
    def update(i):
        ax.clear()
        im = ax.imshow(model.frames[i], cmap='RdBu', vmin=-1, vmax=1)
        if T is not None:
            ax.set_title(f'Ising Model Simulation - T={plot} - Frame {i} / {len(model.frames)-1}')
        else:
            ax.set_title(f'Ising Model Simulation - Frame {i} / {len(model.frames)-1}')
        # We don't need to return the colorbar as it's persistent
        return [im]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(model.frames), interval=100, blit=True)
    
    # Save with a higher quality and more frames per second
    ani.save(f'ising_simulation_{plot}.gif', writer='pillow', fps=10, dpi=100)
    print(f"Animation saved as 'ising_simulation_{plot}.gif'")
    
    plt.close(fig)

@numba.njit(nopython=True)
def calculate_auto_corr(samples: np.ndarray) -> np.ndarray:
    #print(samples)
    num_measurements = len(samples)
    auto_corr = np.zeros(num_measurements)
    max_lag = num_measurements//5
    for k in range(max_lag):
        upper_lim = num_measurements - k
        mean = np.sum(samples[:upper_lim]) / upper_lim
        for i in range(upper_lim):
            auto_corr[k] += samples[i + k] * (samples[i] - mean)
        auto_corr[k] /= upper_lim
    #print(auto_corr)
    #print(auto_corr[0])
    auto_corr /= auto_corr[0]
    return auto_corr

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
    # Clear any existing frames
    model.frames = []
    
    # Save more frames for better animation (but not too many to keep file size reasonable)
    # Aim for about 50-100 frames total
    save_every = max(1, n_iterations // 100)
    
    spins = model.spins
    J = model.J
    beta = model.beta

    n_samples = 200
    n_sample_interval = 1

    magn_list = []
    energy_list = []
    sample_start_index = n_iterations - n_samples * n_sample_interval
    # Add initial state as first frame
    if plot:
        model.frames.append(spins.copy())
    
    up, down, left, right = setup_neighbor_indices(spins.shape[0], spins.shape[1])
    start_time = time.time()

    E1, E2 = 0,0
    M = np.zeros_like(spins)
    M1, M2 = 0,0
    
    spin_samples = []

    # Main simulation loop
    for iter in tqdm(range(n_iterations)):
        # Perform single Metropolis step
        spins = metropolis_step(spins, J, beta, up, down, left, right)
        #spins = wolff_cluster_update(spins, model.temperature, J)
            
        # Save frame every save_every iterations if plotting is enabled
        if plot and iter % save_every == 0:
            model.frames.append(spins.copy())
        
        if iter >= sample_start_index and iter % n_sample_interval == 0:
            magn = calc_magnetization(spins) # sum over all spins / n_sites
            magn_list.append(magn/spins.size)
            ener = calc_hamiltonian(spins, J)
            energy_density = ener / spins.size
            energy_list.append(energy_density)
            E1 += ener
            E2 += ener**2
            M += spins
            M1 += magn
            M2 += magn**2


    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Collected {len(model.frames)} frames for animation")

    model.spins_final = spins
    model.energy_final = np.mean(energy_list)
    model.magnetization_final = np.mean(magn_list)
    model.energy_list = energy_list
    model.magnetization_list = magn_list


    # calculate the specific heat 
    specific_heat = ((E2/n_samples) - (E1*E1/(n_samples**2)))/(model.temperature**2 * spins.size)
    #specific_heat = ((E2/n_samples) - ((E1**2)/(n_samples**2)))/(model.temperature**2)
    model.E1 = E1/(n_samples)
    model.E2 = E2/(n_samples**2)
    model.specific_heat = specific_heat

    # calculate the susceptibility
    M = M / n_samples     # average magnetization per spin * size of system 
    model.M = M/spins.size
    model.susceptibility = ((M2/n_samples) - (M1*M1/(n_samples**2)))/(model.temperature * spins.size)


    # Calculate the autocorrelation time for E
    ener_samples = np.array(model.energy_list)
    model.auto_corr = calculate_auto_corr(ener_samples)
    # print(model.auto_corr)
    # plt.plot(model.auto_corr)
    # plt.show()
            
    if plot:
        animate_ising_model(model, T=round(model.temperature, 2), plot=plot)
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
    output_dir = "output_grid"
    os.makedirs(output_dir, exist_ok=True)
    models = []
    for temp in temps:
        model = IsingModel(size=grid_size, temperature=temp, J=J)
        model = simulate_ising_model(model, n_iterations=n_iterations)
        print(f"Final model energy: {model.energy_final}, Final model magnetization: {model.magnetization_final}")
        models.append(model)
    
    # Extract energy and magnetization data
    energies = np.array([model.energy_final for model in models])
    magnetizations = np.array([model.magnetization_final for model in models])
    
    # Calculate specific heat and susceptibility
    specific_heat = calculate_specific_heat(temps, energies)
    susceptibility = calculate_susceptibility(temps, magnetizations)
    
    # Plot energy, magnetization, specific heat, and susceptibility
    f = plt.figure(figsize=(18, 10))
    sp = f.add_subplot(2, 2, 1)
    sp.scatter(temps, energies, color="IndianRed", marker="o", s=50)
    sp.set_xlabel("Temperature")
    sp.set_ylabel("Energy")
    
    sp = f.add_subplot(2, 2, 2)
    sp.scatter(temps, magnetizations, color="RoyalBlue", marker="o", s=50)
    sp.set_xlabel("Temperature")
    sp.set_ylabel("Magnetization")
    
    sp = f.add_subplot(2, 2, 3)
    sp.scatter(temps, specific_heat, color="ForestGreen", marker="o", s=50)
    sp.set_xlabel("Temperature")
    sp.set_ylabel("Specific Heat")
    
    sp = f.add_subplot(2, 2, 4)
    sp.scatter(temps, susceptibility, color="DarkOrange", marker="o", s=50)
    sp.set_xlabel("Temperature")
    sp.set_ylabel("Susceptibility")
    
    plt.tight_layout()
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
    output_dir = "output_grid"
    os.makedirs(output_dir, exist_ok=True)

    # Run a single simulation
    model = IsingModel(size=100, temperature=0.5, J=1.0)
    model = simulate_ising_model(model, n_iterations=10_000, plot=os.path.join(output_dir, f"{model.temperature}"))
    print("Final model energy: ", model.energy_final, "Final model magnetization: ", model.magnetization_final)

    model = IsingModel(size=100, temperature=2.27, J=1.0)
    model = simulate_ising_model(model, n_iterations=10_000, plot=os.path.join(output_dir, f"{model.temperature}"))
    print("Final model energy: ", model.energy_final, "Final model magnetization: ", model.magnetization_final)

    model = IsingModel(size=100, temperature=4.0, J=1.0)
    model = simulate_ising_model(model, n_iterations=10_000, plot=os.path.join(output_dir, f"{model.temperature}"))
    print("Final model energy: ", model.energy_final, "Final model magnetization: ", model.magnetization_final)


    temps = np.linspace(0.5, 5.0, 50)
    simulate_ising_model_temps(temps, grid_size=30, J=1.0, n_iterations=10_000)