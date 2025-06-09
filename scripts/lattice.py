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

from main_tree import IsingModel, simulate_ising_full
from utils.gen_utils import dimensional_crossover, create_3D_lattice


output_dir = "output/lattice"
os.makedirs(output_dir, exist_ok=True)


sizes = np.arange(10,20,10)
temps = np.linspace(0.1, 6.0, 100)

magnetizations, energies = {}, {}
specific_heat, susceptibility = {}, {}
auto_corr = {}
for size in sizes:
    
    # create the 3D cube
    spins, neighbors = create_3D_lattice(size)
    
    magnetizations[size], energies[size] = [], []
    specific_heat[size] = []
    susceptibility[size] = []
    auto_corr[size] = {}
    
    spins = np.random.choice([-1,1], size=spins.shape[0])
    results = simulate_ising_full(spins, neighbors, temps=temps, J=1.0, n_equilib_steps=3, n_mcmc_steps=100, n_sample_interval=1, step_algorithm="wolff", n_cores=30, n_samples=None)

    for T in results.keys():
        magnetizations[size].append(results[T].avg_magn)
        energies[size].append(results[T].avg_energy)
        specific_heat[size].append(results[T].specific_heat)
        susceptibility[size].append(results[T].susceptibility)

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
#plt.show()




# plot the auto_corr
for size in sizes:
    plt.figure()
    # set log y axis
    #plt.yscale('log')
    # set the y limit to 1 and add ticks 1/e and 1/e^2
    plt.ylim(0.0, 1.0)
    #plt.yticks([0.1,1/np.e,1/np.e**2])
    for T in auto_corr[size].keys():
        plt.plot(auto_corr[size][T], label=f'T={T:.2f}')
    plt.title(f'size={size}')
    plt.legend()
plt.show()

