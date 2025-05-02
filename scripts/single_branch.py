import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
import psutil
import gc 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from branch_sim import MamSimulation
from utils.gen_utils import graph_to_model_format, plot_magn_energy
from main_tree import simulate_ising_model, animate_ising_model, plot_graph
from IsingModel import IsingModel

output_dir = "output_mam"
os.makedirs(output_dir, exist_ok=True)

## set parameters ##
tmax = 300
T = 0.5
J = 1.0
n_equilib_steps = 1_000
n_mcmc_steps = 1_000
n_samples = 30



## Generate the branch network ##
sim = MamSimulation(tmax=tmax)
coordinates, evolve, G = sim.simulate()

nodes, neighbors = graph_to_model_format(G)

## Run the ising model on the branch network ##

model = IsingModel(
    nodes, 
    neighbors,
    temp=T,
    J=J,
    n_equilib_steps=n_equilib_steps,
    n_mcmc_steps=n_mcmc_steps,
    n_samples=n_samples,
    G=G)

model = simulate_ising_model(model)
print(model.total_magn)
print(model.total_energy)

animate_ising_model(model, output_dir=output_dir, save_percent=10)


# Create a denser sampling around the critical region
temps = np.linspace(0.1, 5.0, 100)
temps_specific = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0,3.5,4.0])
temps = np.sort(np.unique(np.concatenate([temps, temps_specific])))

# create a figure with a subplot for each temperature in a grid 4x4
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
axs_index = 0
magnetizations, energies = [], []
for T in temps:
    
    model = IsingModel(nodes, 
                        neighbors,
                        temp=T,
                        J=J,
                        n_equilib_steps=n_equilib_steps,
                        n_mcmc_steps=n_mcmc_steps,
                        n_samples=n_samples,
                        G=G)
    model = simulate_ising_model(model)
    magnetizations.append(model.total_magn)
    energies.append(model.total_energy)
    spins_final = model.spins[-1]

    # plot the final graph with the spins as colors
    if T in temps_specific:
        pos = nx.get_node_attributes(G, 'pos')
        color_map = ["blue" if spin == 1 else "red" for spin in spins_final]
        nx.draw(G, pos=pos, with_labels=False, node_size=10, node_color=color_map, ax=axs[axs_index//4, axs_index%4])
        axs[axs_index//4, axs_index%4].set_title(f"T={T}")
        axs[axs_index//4, axs_index%4].axis('off')
        axs_index += 1
        #plt.show()

    
# add a title to the figue 
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"n_mcmc{n_mcmc_steps}_n_samples{n_samples}_J{J}_spins.png"))
plt.show()
plt.close()

title = f"n_layers=1, n_mcmc={n_mcmc_steps}, n_samples={n_samples}, J={J}"
plot_magn_energy(magnetizations, energies, temps, save_path=os.path.join(output_dir, f"n_mcmc{n_mcmc_steps}_n_samples{n_samples}_J{J}_global_stats.png"), show=True, title=title)

