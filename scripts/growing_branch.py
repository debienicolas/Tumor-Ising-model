"""
Script to:
- Generate a very large branch -> in function of time steps
- Run the Ising model on the growing branch
- Do some mc steps to reach equilibrium, then extend the branch by 
"""
import os
import pickle

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from branch_sim import MamSimulation
from utils.gen_utils import graph_to_model_format
from main_tree import IsingModel, simulate_ising_model, animate_ising_model, plot_graph
from utils.branch_sim_utils import plot_branch_graph, plot_branch_network, model_format_at_time

output_dir = "output_growing"
os.makedirs(output_dir, exist_ok=True)

### Set parameters ###
tmax = 100
t_start = 10

J = 1.0
T = 2.0
n_equilib_steps = 1_000
n_mcmc_steps = 1_000
n_samples = 10
n_sample_interval = 10

### Generate the branch network ###

sim = MamSimulation(tmax=tmax)
coordinates, evolve, G = sim.simulate()

## build a simple plot of nodes in the branch at time points
x = np.arange(t_start, tmax).astype(int)
y = [np.sum(evolve[:i]) for i in x]

plt.figure(figsize=(10, 10))
plt.scatter(x, y, marker='o', color="RoyalBlue")
plt.xlabel("Time")
plt.ylabel("Number of nodes")
plt.savefig(os.path.join(output_dir, "branch_nodes_over_time.png"))
plt.show()

plot_branch_network(coordinates, evolve, save_path=os.path.join(output_dir, "network_structure.png"))
plot_branch_graph(G, os.path.join(output_dir, "network_graph.png"))

nodes, neighbors = model_format_at_time(coordinates, evolve, t_start)


model = IsingModel(
    nodes, neighbors, temp=T, J=J,
    coords=coordinates, evolve=evolve, start_time=t_start,
    n_equilib_steps=n_equilib_steps,
    n_mcmc_steps=n_mcmc_steps,
    n_samples=n_samples,
    n_sample_interval=n_sample_interval
)

model = simulate_ising_model(model)



x = list(model.spins.keys())
magn = []
energy = []
for t in x:
    magn.append(np.mean(np.array(model.total_magn[t])))
    energy.append(np.mean(np.array(model.total_energy[t])))

plt.figure(figsize=(20, 10))
sp = plt.subplot(1, 2, 1)
sp.scatter(x, energy, label='energy', marker='o', color="IndianRed")
sp.set_xlabel("Time")
sp.set_ylabel("Energy")
sp = plt.subplot(1, 2, 2)
sp.scatter(x, magn, label='magnetization', marker='o', color="RoyalBlue")
sp.set_xlabel("Time")
sp.set_ylabel("Magnetization")
plt.savefig(os.path.join(output_dir, f"J{J}_T{T}_magnetization_energy.png"))
plt.show()


#### create an animation of the spins
from PIL import Image
from utils.branch_sim_utils import convert_branch_coords_to_graph
# Convert spin arrays to image format first
image_frames = []
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

# Process each frame
frame_count = 0
for t, spin_array in model.spins.items():
    print(f"Processing frames time step {t}")
    _, neighbors = model_format_at_time(coordinates, evolve, t)
    G = convert_branch_coords_to_graph(coordinates[:np.sum(evolve[:t])], 1.25)
    for i in range(spin_array.shape[0]):
        if i % 100 == 0:
            ax.clear()  # Clear previous frame
            plot_graph(spin_array[i],neighbors, ax=ax, G=G)
        
            ax.set_title(f'Time step {t} - MCMC step {i}/{spin_array.shape[0]} - Temp={T}')
            frame_count += 1
        
            # Render to image
            fig.canvas.draw()
            image = np.array(fig.canvas.buffer_rgba())
            image_frames.append(image)

plt.close(fig)

# Create GIF with PIL
frames_pil = [Image.fromarray(frame) for frame in image_frames]
frames_pil[0].save(
    os.path.join(output_dir, f"J{J}_T{T}_branch_animation.gif"),
    save_all=True,
    append_images=frames_pil[1:],
    duration=100,  # Time between frames in milliseconds (faster)
    loop=0
)
print(f"Animation saved as '{os.path.join(output_dir, f'J{J}_T{T}_branch_animation.gif')}'")


# model = IsingModel(nodes, neighbors, temperature=T, J=J, G=mam_initial.G, branch_sim=mam, current_branch_time=50)
# model = simulate_ising_model(model, n_iterations=n_iter)
# animate_ising_model(model, output_dir=output_dir, T=T)



# # plot the final graph
# ax = plot_graph(model.spins_final, model.neighbors, G=model.G, draw_edges=False)
# ax.set_title(f"T={T}")
# plt.savefig(os.path.join(output_dir, f"final_graph_T={T}.png"))
#plt.show()


# simulating over range of temperatures
# temps = np.linspace(0.1, 5.0, 100)
# magnetization = []
# energy = []
# for T in tqdm(temps):
#     mam_initial = MamSimulation(tmax=50)
#     mam_initial.simulate()
#     mam_initial.plot_network(show=False)
#     mam_initial.convert_to_graph(plot=False)
#     nodes, neighbors = graph_to_model_format(mam_initial.G)

#     print("nodes.shape: ", nodes.shape)
#     print("neighbors.shape: ", neighbors.shape)
#     model = IsingModel(nodes, neighbors, temperature=T, J=J, G=mam_initial.G, branch_sim=mam, current_branch_time=50)
#     model = simulate_ising_model(model, n_iterations=n_iter)
#     magnetization.append(model.magnetization_final)
#     energy.append(model.energy_final)

# plt.figure(figsize=(20, 10))
# sp = plt.subplot(1, 2, 1)
# sp.scatter(temps, energy, label='energy', marker='o', color="IndianRed")
# sp.set_xlabel("Temperature")
# sp.set_ylabel("Energy")
# #plt.legend()
# sp = plt.subplot(1, 2, 2)
# sp.scatter(temps, magnetization, label='magnetization', marker='o', color="RoyalBlue")
# sp.set_xlabel("Temperature")
# sp.set_ylabel("Magnetization")
# #plt.legend()
# plt.savefig(os.path.join(output_dir, f"n_iter{n_iter}_J{J}_magnetization_energy.png"))
# plt.show()