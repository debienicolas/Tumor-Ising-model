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

from gen_mam import MamSimulation
from utils import graph_to_model_format, dimensional_crossover
from main_tree import IsingModel, simulate_ising_model, animate_ising_model, plot_graph


if __name__ == "__main__":
    n_layers = 2
    output_dir = f"output_dim_cross/{n_layers}_layers"
    os.makedirs(output_dir, exist_ok=True)

    T = 4.0
    J = 1.0
    n_iter = 10_000

    # generate the branch
    mam = MamSimulation(tmax=150)
    mam.simulate()
    mam.plot_network(os.path.join(output_dir, "network_structure.png"))
    mam.convert_to_graph(os.path.join(output_dir, "network_graph.png"))
    

    # Stack the branch n times
    stacked_graph = dimensional_crossover(mam.G, n_layers, pos_offset=200)
    with open(os.path.join(output_dir, "stacked_graph.pkl"), "wb") as f:
        pickle.dump(stacked_graph, f)

    # convert the stacked graph to ising model format
    spins, neighbors = graph_to_model_format(stacked_graph)


    ### Run the Ising model on the stacked branch
    model = IsingModel(spins, neighbors, temperature=T, J=J, G=stacked_graph)
    model = simulate_ising_model(model, n_iterations=n_iter)
    animate_ising_model(model, output_dir=output_dir, T=T)
    ### plot the final graph
    ax = plot_graph(model.spins_final,neighbors, G=stacked_graph, draw_edges=False)
    ax.set_title(f"T={T}")
    plt.savefig(os.path.join(output_dir, f"stacked_graph_T={T}.png"))
    plt.show()

    # plot the final spins each layer seperately
    nodes_per_layer = len(stacked_graph.nodes) // stacked_graph.graph["layers"]
    assert len(stacked_graph.nodes) % stacked_graph.graph["layers"] == 0, "The number of nodes must be divisible by the number of layers"

    fig, axs = plt.subplots(1,stacked_graph.graph["layers"], figsize=(20,10))
    for i, ax in enumerate(axs):
        layer_graph = nx.subgraph(stacked_graph, [node for node in stacked_graph.nodes if node[1] == i])
        plot_graph(model.spins_final[i*nodes_per_layer:(i+1)*nodes_per_layer], neighbors[i*nodes_per_layer:(i+1)*nodes_per_layer], G=layer_graph, draw_edges=False, ax=ax)
        ax.set_title(f"Layer {i}")
    plt.savefig(os.path.join(output_dir, f"stacked_graph_layers_T={T}.png"))
    plt.show()

    plt.close()



    ### Run the Ising model on various temperatures and plot the results
    temps = np.linspace(0.1, 5.0, 100)
    magnetization = []
    energy = []
    for T in temps:
        model = IsingModel(spins, neighbors, temperature=T, J=J, G=stacked_graph)
        model = simulate_ising_model(model, n_iterations=n_iter)
        magnetization.append(model.magnetization_final)
        energy.append(model.energy_final)

    plt.figure(figsize=(20, 10))
    sp = plt.subplot(1, 2, 1)
    sp.scatter(temps, energy, label='energy', marker='o', color="IndianRed")
    sp.set_xlabel("Temperature")
    sp.set_ylabel("Energy")
    #plt.legend()
    sp = plt.subplot(1, 2, 2)
    sp.scatter(temps, magnetization, label='magnetization', marker='o', color="RoyalBlue")
    sp.set_xlabel("Temperature")
    sp.set_ylabel("Magnetization")
    #plt.legend()
    plt.savefig(os.path.join(output_dir, f"n_iter{n_iter}_J{J}_magnetization_energy.png"))
    plt.show()
