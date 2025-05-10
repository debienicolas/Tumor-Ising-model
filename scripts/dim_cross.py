"""
Script to:
- Generate a branch
- Stack to branch n times
- Run the Ising model on the stacked branch
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from branch_sim import MamSimulation
from utils.gen_utils import graph_to_model_format, dimensional_crossover, plot_magn_energy
from main_tree import IsingModel, simulate_ising_model, animate_ising_model, plot_graph


n_layers = 2

output_dir = f"output_dim_cross/{n_layers}_layers"
os.makedirs(output_dir, exist_ok=True)


tmax = 150
T = 0.5
J = 1.0
n_equilib_steps = 1_000
n_mcmc_steps = 1_000
n_samples = 50


# generate the branch
mam = MamSimulation(tmax=tmax)
coordinates, evolve, G = mam.simulate()


# Stack the branch n times
stacked_graph = dimensional_crossover(G, n_layers, pos_offset=200)
with open(os.path.join(output_dir, "stacked_graph.pkl"), "wb") as f:
    pickle.dump(stacked_graph, f)

# convert the stacked graph to ising model format
nodes, neighbors = graph_to_model_format(stacked_graph)


### Run the Ising model on the stacked branch
model = IsingModel(nodes, neighbors, temp=T, J=J, G=stacked_graph, n_equilib_steps=n_equilib_steps, n_mcmc_steps=n_mcmc_steps, n_samples=n_samples)
model = simulate_ising_model(model)
animate_ising_model(model, output_dir=output_dir)

### plot the final graph
ax = plot_graph(model.spins[-1],neighbors, G=stacked_graph, draw_edges=False)
ax.set_title(f"T={T}")
plt.savefig(os.path.join(output_dir, f"stacked_graph_T={T}.png"))
plt.show()

# plot the final spins each layer seperately
nodes_per_layer = len(stacked_graph.nodes) // stacked_graph.graph["layers"]
assert len(stacked_graph.nodes) % stacked_graph.graph["layers"] == 0, "The number of nodes must be divisible by the number of layers"

fig, axs = plt.subplots(1,stacked_graph.graph["layers"], figsize=(20,10))
for i, ax in enumerate(axs):
    layer_graph = nx.subgraph(stacked_graph, [node for node in stacked_graph.nodes if node[1] == i])
    plot_graph(model.spins[-1][i*nodes_per_layer:(i+1)*nodes_per_layer], neighbors[i*nodes_per_layer:(i+1)*nodes_per_layer], G=layer_graph, draw_edges=False, ax=ax)
    ax.set_title(f"Layer {i}")
# set title of the figure
fig.suptitle(f"T={T}")
plt.savefig(os.path.join(output_dir, f"stacked_graph_layers_T={T}.png"))
plt.show()

plt.close()



### Run the Ising model on various temperatures and plot the results
temps = np.linspace(0.1, 5.0, 100)
magnetization = []
energy = []
for T in temps:
    model = IsingModel(nodes, neighbors, temp=T, J=J, G=stacked_graph, n_equilib_steps=n_equilib_steps, n_mcmc_steps=n_mcmc_steps, n_samples=n_samples)
    model = simulate_ising_model(model)
    magnetization.append(model.total_magn)
    energy.append(model.total_energy)

title = f"n_layers={n_layers}, n_mcmc={n_mcmc_steps}, n_samples={n_samples}"
plot_magn_energy(magnetization, energy, temps, save_path=os.path.join(output_dir, f"n_mcmc{n_mcmc_steps}_n_samples{n_samples}_J{J}_global_stats.png"), show=True, title=title)