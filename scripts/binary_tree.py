"""
Use this script to run the ising model on a binary tree for various depths
"""
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
import psutil
import gc 
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import wandb


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from IsingModel import IsingModel
from main_tree import simulate_ising_model


# output_dir = "output/binary_tree"
# os.makedirs(output_dir, exist_ok=True)
# output_dir = os.path.join(output_dir, f"{int(time.time())}")
# os.makedirs(output_dir, exist_ok=True)


def create_nodes_and_neighbors(depth:int):
    # Create a binary tree using NetworkX
    G = nx.balanced_tree(2, depth-1)  # 2 children per node, depth-1 levels
    
    # Initialize nodes array (all nodes start with value 1)
    nodes = np.ones(len(G))
    
    # Initialize neighbors array
    neighbors = np.zeros((len(G), 3))  # Changed to 3 to store [parent, child1, child2]
    neighbors.fill(-1)  # Default to no neighbors
    
    # For each node, get its neighbors (both children and parent)
    for node in sorted(G.nodes()):
        # Get all neighbors (both children and parent)
        all_neighbors = sorted(list(G.neighbors(node)))
        
        # For root node (node 0), it only has children
        if node == 0:
            if len(all_neighbors) == 2:
                neighbors[node] = np.array([-1, all_neighbors[0], all_neighbors[1]])
        # For other nodes, they have one parent and potentially two children
        else:
            # Find the parent (it's the neighbor with smaller index)
            parent = min(all_neighbors)
            # Find the children (they're the neighbors with larger indices)
            children = [n for n in all_neighbors if n > node]
            
            # Store parent and children
            if len(children) == 2:
                neighbors[node] = np.array([parent, children[0], children[1]])
            elif len(children) == 1:
                neighbors[node] = np.array([parent, children[0], -1])
            else:
                neighbors[node] = np.array([parent, -1, -1])
    
    # turn neighbors into list of ints
    neighbors = neighbors.astype(np.int32)
    
    # Debug print
    # print("Tree structure:")
    # for node in sorted(G.nodes()):
    #     print(f"Node {node}: [parent, child1, child2] = {neighbors[node]}")
    
    return nodes, neighbors

#print(create_nodes_and_neighbors(3))


sizes = np.arange(6,11,2)
sizes = np.array([5,6,7])
temps = np.linspace(0.1, 5.0, 200)


#temps = np.linspace(0.5, 2.5,10)
#temps = np.array([0.5,1.0,1.5,2.0,2.27,3.0])

id = f"depth_{sizes[0]}_to_{sizes[-1]}"

b = 2
# first critical temperature
beta_c_0 = 0.5*np.log((b+1)/(b-1))
t_c_0 = 1/beta_c_0

# second critical temperature 
beta_c_1 = 0.5*np.log((np.sqrt(b)+1)/(np.sqrt(b)-1))
t_c_1 = 1/beta_c_1

##### 
n_equilib_steps = 50_000
n_mcmc_steps = 50_000
n_sample_interval = 5
n_samples = 1000



size = 6

# set the tag of the runs to the experiment name
tag = "step_algorithm"

algos = ["wolff", "metropolis", "glauber"]

for algo in algos:

    n_samples = 1000
    if algo == "wolff":
        n_equilib_steps = 10_000
        n_mcmc_steps = 10_000
        n_sample_interval = 10
    elif algo == "metropolis" or algo == "glauber":
        n_equilib_steps = 50_000
        n_mcmc_steps = 50_000
        n_sample_interval = 10

    config = {
        "structure": "regular_tree",
        "depth": size,
        "temps": temps,
        "b": b,
        "n_equilib_steps": n_equilib_steps,
        "n_mcmc_steps": n_mcmc_steps,
        "n_sample_interval": n_sample_interval,
        "n_samples": n_samples,
        "step_algorithm": algo
    }

    with wandb.init(config=config, project="ising_model", tags=[tag]) as run:

        nodes, neighbors = create_nodes_and_neighbors(size)
        magnetizations,energies = [],[]
        specific_heat, susceptibility = [],[]
        for T in temps:
            nodes = np.random.choice([-1,1], size=nodes.shape[0])
            model = IsingModel(nodes=nodes,
                               neighbors=neighbors,
                               temp=T,
                               J=1.0,
                               n_equilib_steps=n_equilib_steps,
                               n_mcmc_steps=n_mcmc_steps,n_sample_interval=n_sample_interval,
                               n_samples=n_samples)
            model = simulate_ising_model(model)

            magnetizations.append(model.total_magn)
            energies.append(model.total_energy)
            specific_heat.append(model.specific_heat)
            susceptibility.append(model.susceptibility)

            run.log({
                "magnetization": model.total_magn,
                "energy": model.total_energy,
                "specific_heat": model.specific_heat,
                "susceptibility": model.susceptibility,
                "temp": T
            })

            #auto_corr[size][T] = model.auto_corr        
        # smooth the data
        sm_magnetization = gaussian_filter1d(magnetizations, sigma=4)
        sm_energy = gaussian_filter1d(energies, sigma=4)
        sm_specific_heat = gaussian_filter1d(specific_heat, sigma=4)
        sm_susceptibility = gaussian_filter1d(susceptibility, sigma=4)
        for i, T in enumerate(temps):
            run.log({
                "sm_magnetization": sm_magnetization[i],
                "sm_energy": sm_energy[i],
                "sm_specific_heat": sm_specific_heat[i],
                "sm_susceptibility": sm_susceptibility[i],
                "temp": T
            })

        


