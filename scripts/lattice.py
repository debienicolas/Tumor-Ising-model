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

from main import IsingModel, simulate_ising_model


output_dir = "output/lattice"
os.makedirs(output_dir, exist_ok=True)


sizes = np.arange(10,50,10)
temps = np.linspace(0.1, 5.0, 200)

magnetizations, energies = {}, {}
specific_heat, susceptibility = {}, {}
for size in sizes:
    magnetizations[size], energies[size] = [], []
    specific_heat[size] = []
    susceptibility[size] = []
    for T in temps:
        model = IsingModel(size=size, temperature=T)
        model = simulate_ising_model(model, n_iterations=5_000)
        magnetizations[size].append(model.magnetization_final)
        energies[size].append(model.energy_final)
        specific_heat[size].append(model.specific_heat)
        susceptibility[size].append(model.susceptibility)

    # smooth the data
    sigma = 4
    magnetizations[size] = gaussian_filter1d(magnetizations[size], sigma=sigma)
    energies[size] = gaussian_filter1d(energies[size], sigma=sigma)
    specific_heat[size] = gaussian_filter1d(specific_heat[size], sigma=sigma)
    susceptibility[size] = gaussian_filter1d(susceptibility[size], sigma=sigma)


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for size in sizes:    

    # first axis should be the energy    
    axs[0,0].plot(temps, energies[size], label=f'size={size}')
    # second axis should be the magnetization
    axs[0,1].plot(temps, magnetizations[size], label=f'size={size}')
    # third axis should be the specific heat
    axs[1,0].plot(temps, specific_heat[size], label=f'size={size}')
    # fourth axis should be the susceptibility
    axs[1,1].plot(temps, susceptibility[size], label=f'size={size}')

    # find the critical temperature - temp where spec heat is max
    critical_temp = temps[np.argmax(np.array(specific_heat[size]))]
    #axs[1,0].axvline(critical_temp, color='black', linestyle='--', label=f'critical temp={critical_temp:.2f}')
    #axs[1,1].axvline(critical_temp, color='black', linestyle='--', label=f'critical temp={critical_temp:.2f}')


axs[0,0].set_ylabel('Energy')
axs[0,1].set_ylabel('Magnetization')

axs[1,0].set_xlabel('Temperature')
axs[1,0].set_ylabel('Specific Heat')

axs[1,1].set_xlabel('Temperature')
axs[1,1].set_ylabel('Susceptibility')
axs[0,0].legend()
axs[0,1].legend()
axs[1,0].legend()
axs[1,1].legend()
# set a title for the plot
plt.suptitle('Lattice')
plt.tight_layout()

plt.savefig(os.path.join(output_dir, f'lattice_overview.png'))
plt.show()


