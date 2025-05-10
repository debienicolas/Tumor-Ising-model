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
from main_tree import simulate_ising_model, animate_ising_model, plot_graph
from IsingModel import IsingModel

output_dir = "output/single_branch"
os.makedirs(output_dir, exist_ok=True)

## set parameters ##
tmax = 100
T = 0.5
J = 1.0
n_equilib_steps = 5_000
n_mcmc_steps = 1_000
n_samples = 1_000



temps = np.linspace(0.01, 5.0, 200)
tmax_list = np.array([100,125,150,175])

magnetizations, energies = {}, {}
specific_heat, susceptibility = {}, {}
for tmax in tmax_list:
    magnetizations[tmax], energies[tmax], specific_heat[tmax], susceptibility[tmax] = [], [], [], []
    sim = MamSimulation(tmax=tmax)
    coordinates, evolve, G = sim.simulate()
    nodes, neighbors = graph_to_model_format(G)
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
        magnetizations[tmax].append(model.total_magn)
        energies[tmax].append(model.total_energy)
        specific_heat[tmax].append(model.specific_heat)
        susceptibility[tmax].append(model.susceptibility)
    
    # smooth the data
    sigma = 4
    magnetizations[tmax] = gaussian_filter1d(magnetizations[tmax], sigma=sigma)
    energies[tmax] = gaussian_filter1d(energies[tmax], sigma=sigma)
    specific_heat[tmax] = gaussian_filter1d(specific_heat[tmax], sigma=sigma)
    susceptibility[tmax] = gaussian_filter1d(susceptibility[tmax], sigma=sigma)


fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# plot the specific heats for each tmax
for tmax in tmax_list:
    axs[0,0].plot(temps, energies[tmax], label=f'tmax={tmax}')
    axs[0,1].plot(temps, magnetizations[tmax], label=f'tmax={tmax}')
    axs[1,0].plot(temps, specific_heat[tmax], label=f'tmax={tmax}')
    axs[1,1].plot(temps, susceptibility[tmax], label=f'tmax={tmax}')

axs[0,0].set_ylabel('Energy')

axs[0,1].set_ylabel('Magnetization')

axs[1,0].set_xlabel('Temperature')
axs[1,0].set_ylabel('Specific Heat')

axs[1,1].set_xlabel('Temperature')
axs[1,1].set_ylabel('Susceptibility')

#axs[0,0].legend()
axs[0,1].legend()
#axs[1,0].legend()
#axs[1,1].legend()

plt.suptitle('Branch Ising Model')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "branch_overview.png"))
plt.show()

    
# add a title to the figue 
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, f"n_mcmc{n_mcmc_steps}_n_samples{n_samples}_J{J}_spins.png"))
# plt.show()
# plt.close()

#title = f"n_layers=1, n_mcmc={n_mcmc_steps}, n_samples={n_samples}, J={J}"
#plot_magn_energy(magnetizations, energies, temps, save_path=os.path.join(output_dir, f"n_equib_{n_equilib_steps}n_mcmc{n_mcmc_steps}_n_samples{n_samples}_J{J}_global_stats.png"), show=True, title=title)

