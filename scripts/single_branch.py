import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
import psutil
import gc 
from scipy.ndimage import gaussian_filter1d
import time
import mlflow

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from branch_sim import MamSimulation
from utils.gen_utils import graph_to_model_format, plot_magn_energy
from main_tree import simulate_ising_model, animate_ising_model, plot_graph
from IsingModel import IsingModel
from utils.gen_utils import autocorr

output_dir = "output/single_branch"
os.makedirs(output_dir, exist_ok=True)

# create a subdir with time in seconds
output_dir = os.path.join(output_dir, str(int(time.time())))
os.makedirs(output_dir, exist_ok=True)

# create the time profiling log file
time_profiling = os.path.join(output_dir, "time_profiling.txt")
with open(time_profiling, 'w') as f:
    f.write("Profiling Results\n")
    f.write("=" * 50 + "\n\n")

def log_time(message, prof_file):
    with open(prof_file, 'a') as f:
        f.write(f"{message}\n")

## set parameters ##
tmax = 100
T = 0.5
J = 1.0
n_equilib_steps = 1_000
n_mcmc_steps = 1_000
n_samples = 100
n_sample_interval = 10
seed = 43
step_algorithm = "wolff"

prob_branch = 0.03

#tmax_list = np.array([250,300,350,375])
tmax_list = np.array([75,100,125,150])
tmax_list = np.array([75,100,125])
tmax_list = np.array([100])
#temps = np.linspace(0.01, 2.5, 100)
temps = np.linspace(0.01, 1.5, 50)
temps = np.linspace(0.01,5.0,100)


# log the parameters to the time profiling file
with open(time_profiling, 'a') as f:
    f.write(f"Parameters:\n")
    f.write(f"seed: {seed}\n")
    f.write(f"tmax: {tmax}\n")
    f.write(f"T: {T}\n")
    f.write(f"J: {J}\n")
    f.write(f"n_equilib_steps: {n_equilib_steps}\n")
    f.write(f"n_mcmc_steps: {n_mcmc_steps}\n")
    f.write(f"n_samples: {n_samples}\n")
    f.write(f"n_sample_interval: {n_sample_interval}\n")
    f.write(f"tmax_list: {tmax_list}\n")
    f.write(f"temps: {temps}\n")
    f.write(f"step_algorithm: {step_algorithm}\n")
    f.write(f"prob_branch: {prob_branch}\n")


log_time("", time_profiling)
log_time("="*50, time_profiling)



magnetizations, energies = {}, {}
specific_heat, susceptibility = {}, {}
sm_magnetizations, sm_energies, sm_specific_heat, sm_susceptibility = {}, {}, {}, {}

energy_samples = {}
for tmax in tmax_list:
    total_start_time = time.time()
    magnetizations[tmax], energies[tmax], specific_heat[tmax], susceptibility[tmax] = [], [], [], []
    sm_magnetizations[tmax], sm_energies[tmax], sm_specific_heat[tmax], sm_susceptibility[tmax] = [], [], [], []
    energy_samples[tmax] = {}
    sim = MamSimulation(tmax=tmax, seed=seed, prob_branch=prob_branch)
    coordinates, evolve, G = sim.simulate()
    nodes, neighbors = graph_to_model_format(G)
    individ_times = []
    for T in temps:
        temp_start_time = time.time()
        nodes = np.random.choice([-1,1], size=nodes.shape[0])
        model = IsingModel(nodes, 
                            neighbors,
                            temp=T,
                            J=J,
                            n_equilib_steps=n_equilib_steps,
                            n_mcmc_steps=n_mcmc_steps,
                            n_samples=n_samples,
                            G=G,
                            n_sample_interval=n_sample_interval,
                            step_algorithm=step_algorithm)
        model = simulate_ising_model(model)
        magnetizations[tmax].append(model.avg_magn)
        energies[tmax].append(model.avg_energy)
        energy_samples[T] = model.energy_samples
        specific_heat[tmax].append(model.specific_heat)
        susceptibility[tmax].append(model.susceptibility)
        temp_end_time = time.time()
        #log_time(f"Size:{tmax} T = {T} completed in {temp_end_time - temp_start_time:.2f} seconds", time_profiling)
        individ_times.append(temp_end_time - temp_start_time)
    total_end_time = time.time()
    log_time(f"tmax {tmax} completed in {total_end_time - total_start_time:.2f} seconds", time_profiling)
    np.array(individ_times)
    log_time(f"Average time per temperature: {np.mean(individ_times):.2f} seconds", time_profiling)
    # log the amount of nodes/sites
    log_time(f"Number of nodes: {nodes.shape[0]}", time_profiling)
    log_time("="*50, time_profiling)
    
    
    # smooth the data
    sigma = 4
    
    sm_magnetizations[tmax] = gaussian_filter1d(magnetizations[tmax], sigma=sigma)
    sm_energies[tmax] = gaussian_filter1d(energies[tmax], sigma=sigma)
    sm_specific_heat[tmax] = gaussian_filter1d(specific_heat[tmax], sigma=sigma)
    sm_susceptibility[tmax] = gaussian_filter1d(susceptibility[tmax], sigma=sigma)



fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# plot the specific heats for each tmax - unsmoothened
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
plt.savefig(os.path.join(output_dir, f"branch_overview_tmax{tmax_list[0]}-{tmax_list[-1]}_unsmoothened.png"))
plt.show()

## plot the smoothened data
# in this figure also plot the:
# derivative of the energy for the specific heat
# derivative of the magnetization for the susceptibility

fig, axs = plt.subplots(3, 2, figsize=(20, 15))
for tmax in tmax_list:
    axs[0,0].plot(temps, sm_energies[tmax], label=f'tmax={tmax}')
    axs[0,1].plot(temps, sm_magnetizations[tmax], label=f'tmax={tmax}')

    # find the max of the specific heat and susceptibility
    max_specific_heat = np.max(sm_specific_heat[tmax])
    max_susceptibility = np.max(sm_susceptibility[tmax])
    axs[1,0].plot(temps, sm_specific_heat[tmax], label=f'tmax={tmax}')
    axs[1,1].plot(temps, sm_susceptibility[tmax], label=f'tmax={tmax}')
    axs[1,0].axvline(temps[np.argmax(sm_specific_heat[tmax])], color='red', linestyle='--')
    axs[1,1].axvline(temps[np.argmax(sm_susceptibility[tmax])], color='red', linestyle='--')

    # calculate the derivative of the energy for the specific heat
    dE_dT = np.gradient(sm_energies[tmax], temps)
    # calculate the derivative of the magnetization for the susceptibility
    dM_dT = np.gradient(sm_magnetizations[tmax], temps)

    axs[2,0].plot(temps, dE_dT, label=f'tmax={tmax}')
    axs[2,1].plot(temps, dM_dT, label=f'tmax={tmax}')

axs[0,0].set_ylabel('Energy')

axs[0,1].set_ylabel('Magnetization')

axs[1,0].set_xlabel('Temperature')
axs[1,0].set_ylabel('Specific Heat - fluct.')

axs[1,1].set_xlabel('Temperature')
axs[1,1].set_ylabel('Susceptibility - fluct.')

axs[2,0].set_xlabel('Temperature')
axs[2,0].set_ylabel('Specific Heat - deriv.')

axs[2,1].set_xlabel('Temperature')
axs[2,1].set_ylabel('Susceptibility - deriv.')

axs[0,1].legend()


plt.suptitle('Branch Ising Model')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"branch_overview_tmax{tmax_list[0]}-{tmax_list[-1]}_smoothened.png"))
plt.show()




# create spacec samples taken from temps
temps = np.random.choice(temps, size=5)
colors = ["red", "blue", "green", "yellow", "purple", "orange"]
init_style = "random"
# energy autocorrelation
for tmax in tmax_list:
    for i, T in enumerate(temps):
        corr = autocorr(energy_samples[tmax][T])
        corr = corr[:500]
        plt.plot(corr, color=colors[i], label=f'T={T}')
    plt.title(f'Energy autocorrelation for tmax={tmax}')
    plt.legend()
    plt.show()