"""
Use this script to conver the lattice representation to using nodes and neigbors
"""

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

#from main import IsingModel, simulate_ising_model
from IsingModel import IsingModel
from main_tree import simulate_ising_model
from utils.gen_utils import create_lattice

output_dir = "output/testing"
os.makedirs(output_dir, exist_ok=True)



sizes = np.arange(10,30,10)
temps = np.linspace(0.1, 5.0, 200)
#temps = np.linspace(0.5, 2.5,10)
#temps = np.array([0.5,1.0,1.5,2.0,2.27,3.0])

magnetizations, energies = {}, {}
sm_magnetizations, sm_energies = {}, {}
specific_heat, susceptibility = {}, {}
sm_specific_heat, sm_susceptibility = {}, {}
auto_corr = {}
for size in tqdm(sizes):
    magnetizations[size], energies[size] = [], []
    specific_heat[size] = []
    susceptibility[size] = []
    auto_corr[size] = {}
    sm_magnetizations[size], sm_energies[size] = [], []
    sm_specific_heat[size] = []
    sm_susceptibility[size] = []
    for T in temps:
        nodes, neighbors = create_lattice(size)
        nodes = np.random.choice([-1, 1], size=nodes.size)
        model = IsingModel(nodes=nodes,
                            neighbors=neighbors,
                            temp=T,
                            J=1.0,
                            n_equilib_steps=500,
                            n_mcmc_steps=500,
                            n_sample_interval=1,
                            n_samples=500,
                            step_algorithm="metropolis")
        model = simulate_ising_model(model)
        magnetizations[size].append(model.avg_magn)
        energies[size].append(model.avg_energy)
        specific_heat[size].append(model.specific_heat)
        susceptibility[size].append(model.susceptibility)
    #auto_corr[size][T] = model.auto_corr
    # smooth the data
    sigma = 4
    sm_magnetizations[size] = gaussian_filter1d(magnetizations[size], sigma=sigma)
    sm_energies[size] = gaussian_filter1d(energies[size], sigma=sigma)
    sm_specific_heat[size] = gaussian_filter1d(specific_heat[size], sigma=sigma)
    sm_susceptibility[size] = gaussian_filter1d(susceptibility[size], sigma=sigma)


fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for size in sizes:    

    # first axis should be the energy    
    axs[0,0].plot(temps, sm_energies[size], label=f'size={size}')
    # second axis should be the magnetization
    axs[0,1].plot(temps, sm_magnetizations[size], label=f'size={size}')
    # third axis should be the specific heat
    axs[1,0].plot(temps, sm_specific_heat[size], label=f'size={size}')
    # fourth axis should be the susceptibility
    axs[1,1].plot(temps, sm_susceptibility[size], label=f'size={size}')

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
    axs[0,1].legend()

    # set a title for the plot
    plt.suptitle('Lattice')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f'lattice_overview_smoothed.png'))


fig, axs = plt.subplots(2, 2, figsize=(15, 10))
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
    axs[0,1].legend()

    # set a title for the plot
    plt.suptitle('Lattice')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f'lattice_overview.png'))



    plt.show()