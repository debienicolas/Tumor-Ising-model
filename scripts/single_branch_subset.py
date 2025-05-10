import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
import psutil
import gc 
from scipy.ndimage import gaussian_filter1d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from branch_sim import MamSimulation
from utils.gen_utils import graph_to_model_format, plot_magn_energy
from utils.branch_sim_utils import plot_branch_network, plot_branch_graph
from main_tree import simulate_ising_model, animate_ising_model, plot_graph
from IsingModel import IsingModel
from main_tree import calc_hamiltonian, calc_magnetization

output_dir = "output/single_branch_subset"
os.makedirs(output_dir, exist_ok=True)

## set parameters ##
tmax = 100
T = 0.5
J = 1.0
n_equilib_steps = 5_000
n_mcmc_steps = 1_000
n_samples = 1_000

temps = np.linspace(0.05, 3.0, 100)
tmax_list = np.array([100,125,150,175])


# define the square subset based on the first graph
sim = MamSimulation(tmax=tmax_list[0])
coordinates, evolve, G = sim.simulate()

width, height = np.max(coordinates[:,0]) - np.min(coordinates[:,0]), np.max(coordinates[:,1]) - np.min(coordinates[:,1])
center_x, center_y = np.min(coordinates[:,0]) + width/2, np.min(coordinates[:,1]) + height/2
square_width, square_height = 0.5*width, 0.8*height
square_x_min, square_x_max = center_x - square_width/2, center_x + square_width/2
square_y_min, square_y_max = center_y - square_height/2, center_y + square_height/2


if False:
    fig, axs = plt.subplots(1,2, figsize=(20,10))
    # plot the square on top of the smallest branch
    plot_branch_network(coordinates, evolve, end_time=tmax_list[0], show=False, ax=axs[0])
    axs[0].plot([square_x_min, square_x_max, square_x_max, square_x_min, square_x_min],
            [square_y_min, square_y_min, square_y_max, square_y_max, square_y_min],
            color='red')

    # color the nodes outside the square red
    colors = np.zeros_like(coordinates[:,0])
    colors[coordinates[:,0] < square_x_min] = 1
    colors[coordinates[:,0] > square_x_max] = 1
    colors[coordinates[:,1] < square_y_min] = 1
    colors[coordinates[:,1] > square_y_max] = 1
    colors = ["red" if c == 1 else "blue" for c in colors]
    plot_branch_graph(G, show=False, node_colors=colors, ax=axs[1])
    plt.show()
    plt.close()

def get_mask(coordinates:np.ndarray, square_x_min:float, square_x_max:float, square_y_min:float, square_y_max:float) -> np.ndarray:
    """
    Create a boolean mask with True for the nodes inside the square
    """
    mask = np.ones(coordinates[:,0].shape[0], dtype=bool)
    mask[coordinates[:,0] < square_x_min] = False
    mask[coordinates[:,0] > square_x_max] = False
    mask[coordinates[:,1] < square_y_min] = False
    mask[coordinates[:,1] > square_y_max] = False
    return mask

magnetizations, energies = {}, {}
specific_heat, susceptibility = {}, {}
for tmax in tmax_list:
    magnetizations[tmax], energies[tmax], specific_heat[tmax], susceptibility[tmax] = [], [], [], []
    sim = MamSimulation(tmax=tmax)
    coordinates, evolve, G = sim.simulate()
    nodes, neighbors = graph_to_model_format(G)
    square_mask = get_mask(coordinates, square_x_min, square_x_max, square_y_min, square_y_max)
    square_mask = np.tile(square_mask, (n_samples, 1))
    print("Mask sum", square_mask.sum())
    for T in temps:
        nodes = np.ones(nodes.shape[0])
        model = IsingModel(nodes, 
                            neighbors,
                            temp=T,
                            J=J,
                            n_equilib_steps=n_equilib_steps,
                            n_mcmc_steps=n_mcmc_steps,
                            n_samples=n_samples,
                            G=G)
        model = simulate_ising_model(model)
        spins = model.spins
        
        spins = spins[square_mask].reshape(n_samples,-1)
        print("Spins shape", spins.shape)
        relevant_spins_size = spins.shape[1]
        print("Relevant spins size", relevant_spins_size)
        magnetization = np.zeros(spins.shape[0])
        energy = np.zeros(spins.shape[0])
        E1, E2 = 0, 0
        M1, M2 = 0, 0
        for i in range(spins.shape[0]):
            magnetization[i] = calc_magnetization(spins[i,:])
            energy[i] = calc_hamiltonian(spins[i,:], neighbors, J)
            E1 += energy[i]
            E2 += energy[i]**2
            M1 += magnetization[i]
            M2 += magnetization[i]**2
        magnetizations[tmax].append(np.mean(magnetization) / relevant_spins_size)
        energies[tmax].append(np.mean(energy) / relevant_spins_size)   
        numerator = (E2 / n_samples) - ((E1**2)/(n_samples**2))
        denominator = T**2 * relevant_spins_size

        specific_heat[tmax].append(numerator/denominator)

        numerator = (M2 / n_samples) - ((M1**2)/(n_samples**2))
        denominator = T * relevant_spins_size
        susceptibility[tmax].append(numerator/denominator)

    
    # smooth the data
    sigma = 4
    magnetizations[tmax] = gaussian_filter1d(magnetizations[tmax], sigma=sigma)
    energies[tmax] = gaussian_filter1d(energies[tmax], sigma=sigma)
    specific_heat[tmax] = gaussian_filter1d(specific_heat[tmax], sigma=sigma)
    susceptibility[tmax] = gaussian_filter1d(susceptibility[tmax], sigma=sigma)


fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# plot the specific heats for each tmax
for tmax in tmax_list:
    axs[0,0].plot(temps, energies[tmax], label=f'tmax={tmax}')
    axs[0,1].plot(temps, magnetizations[tmax], label=f'tmax={tmax}')
    axs[1,0].plot(temps, specific_heat[tmax], label=f'tmax={tmax}')
    axs[1,1].plot(temps, susceptibility[tmax], label=f'tmax={tmax}')

axs[0,0].set_ylabel('Energy')
axs[0,0].set_xlabel('Temperature')
axs[0,0].set_title('Energy')

axs[0,1].set_ylabel('Magnetization')
axs[0,1].set_xlabel('Temperature')
axs[0,1].set_title('Magnetization')

axs[1,0].set_xlabel('Temperature')
axs[1,0].set_ylabel('Specific Heat')
axs[1,0].set_title('Specific Heat')

axs[1,1].set_xlabel('Temperature')
axs[1,1].set_ylabel('Susceptibility')
axs[1,1].set_title('Susceptibility')

axs[0,0].legend()
axs[0,1].legend()
axs[1,0].legend()
axs[1,1].legend()

plt.suptitle('Branch subgraph Ising Model')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "branch_subgraph_overview.png"))
plt.show()

