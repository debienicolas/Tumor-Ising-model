import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import wandb
import time
import os
from scipy.ndimage import gaussian_filter1d


from utils.gen_utils import create_lattice, graph_to_model_format, autocorr
from IsingModel import IsingModel
from main_tree import simulate_ising_model, simulate_ising_full
from branch_sim import MamSimulation
from utils.gen_utils import graph_to_model_format

def run_branch():
    config = wandb.config
    
    # fix the issue with using numpy arrays in wandb
    if config.get("temps") is None:
        temps = np.linspace(config["temps_min"], config["temps_max"], config["temps_num"]).tolist()
    else:
        temps = config["temps"]
    
    # fix the issue with cores in wandb
    if config.get("cores") == 0:
        n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        print(f"Using {n_cores} cores from slurm environment")
    else:
        n_cores = config["cores"]

    # generate the branch
    start_time = time.time()
    sim = MamSimulation(tmax=config["tmax"], seed=[config["seed"]], prob_branch=config["prob_branch"], fav=config["fav"], fchem=config["fchem"])
    coords, evolve, G = sim.simulate()
    nodes, neighbors = graph_to_model_format(G)
    end_time = time.time()
    branch_gen_time = end_time - start_time
    print(f"Branch generation time: {branch_gen_time} seconds")

    # log some global graph properties
    wandb.log({
        "n_nodes": nodes.size,
        "branch_gen_time": branch_gen_time
    })

    # run the simulation
    start_time = time.time()
    nodes = np.random.choice([-1, 1], size=nodes.size)
    models = simulate_ising_full(
        nodes=nodes,
        neighbors=neighbors,
        J=1.0,
        n_equilib_steps=config["n_equilib_steps"],
        n_mcmc_steps=config["n_mcmc_steps"],
        n_samples=config["n_samples"],
        n_sample_interval=config["n_sample_interval"],
        temps=temps,
        step_algorithm=config["step_algorithm"],
        n_cores=n_cores,
    )
    
    mag_list = []
    energy_list = []
    for t in models.keys():
        print(f"Logging temp: {t}")
        wandb.log({
            "temp": t,
            "magnetization": models[t].avg_magn,
            "energy": models[t].avg_energy,
            "specific_heat": models[t].specific_heat,
            "susceptibility": models[t].susceptibility,
        })
        mag_list.append(models[t].avg_magn)
        energy_list.append(models[t].avg_energy)
    
    # smoothen the magnetization and energy curves
    mag_list = np.array(mag_list)
    energy_list = np.array(energy_list)
    
    smooth_mag = gaussian_filter1d(mag_list, sigma=4)
    smooth_energy = gaussian_filter1d(energy_list, sigma=4)
    
    # calculate the derivative of the smoothed curves
    dmag_dt = -np.gradient(smooth_mag, np.array(temps))
    denergy_dt = np.gradient(smooth_energy, np.array(temps))
    
    # log the smoothed curves and derivatives
    for i in range(len(temps)):
        wandb.log({
            f"sm_magnetization": smooth_mag[i],
            f"sm_energy": smooth_energy[i],
            f"dm_dt": dmag_dt[i],
            f"de_dt": denergy_dt[i],
            f"temp": config["temps"][i]
        })

    end_time = time.time()

    sim_time = end_time - start_time
    print(f"Simulation time: {sim_time} seconds")
    wandb.log({
        "sim_time": sim_time
    })
    
    

if __name__ == "__main__":
    run_branch()