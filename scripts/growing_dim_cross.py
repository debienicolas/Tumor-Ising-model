"""
Script to:
- Generate a branch
- stack the branch n_layers times
- Run the Ising model on growing stacked branch
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
from utils.branch_sim_utils import convert_branch_coords_to_graph, model_format_at_time

n_layers = 3

output_dir = f"output_growing_dim_cross/{n_layers}_layers"
os.makedirs(output_dir, exist_ok=True)

t_start = 10
tmax = 100
T = 4.0
J = 1.0
n_equilib_steps = 1_000
n_mcmc_steps = 1_000
n_samples = 10


## generate the final branch
mam = MamSimulation(tmax=tmax)
coordinates, evolve, G = mam.simulate()

coords_init = coordinates[:np.sum(evolve[:t_start])]
G_stacked = convert_branch_coords_to_graph(coords_init, dim_cross=n_layers)
print(G_stacked)
## Stack the branch n_layers times
# stacked_graph = dimensional_crossover(G_stacked, n_layers, pos_offset=200)
# with open(os.path.join(output_dir, "stacked_graph.pkl"), "wb") as f:
#     pickle.dump(stacked_graph, f)

nodes, neighbors = graph_to_model_format(G_stacked)
print(nodes.shape, neighbors.shape)

### Run the Ising model on the stacked branch
model = IsingModel(nodes, neighbors, temp=T, J=J, G=G_stacked, n_equilib_steps=n_equilib_steps, n_mcmc_steps=n_mcmc_steps, n_samples=n_samples,
                   coords=coordinates, evolve=evolve, start_time=t_start,
                   dim_cross=n_layers)
model = simulate_ising_model(model)
#animate_ising_model(model, output_dir=output_dir)

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
    _, neighbors = model_format_at_time(coordinates, evolve, t, dim_cross=n_layers)
    G = convert_branch_coords_to_graph(coordinates[:np.sum(evolve[:t])], 1.25, dim_cross=n_layers)
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
    duration=50,  # Time between frames in milliseconds (faster)
    loop=0
)
print(f"Animation saved as '{os.path.join(output_dir, f'J{J}_T{T}_branch_animation.gif')}'")
