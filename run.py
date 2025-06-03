import os

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import wandb
import time

from utils.gen_utils import create_lattice, graph_to_model_format, autocorr
from IsingModel import IsingModel
from main_tree import simulate_ising_model, simulate_ising_full
from branch_sim import MamSimulation
from utils.gen_utils import graph_to_model_format

from wandb_run import run_branch

### set the amount of cores to use from slurm environment
n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
print(f"Using {n_cores} cores")



def single_run():
    config = {
        "structure": "branch",
        "tmax": 100,
        "temps": np.linspace(0.01, 5.0, 200).tolist(),
        "n_equilib_steps": 5,
        "n_mcmc_steps": 500,
        "n_sample_interval": 1,
        "n_samples": None,
        "step_algorithm": "wolff",
        "fav": -0.1,
        "fchem": 0.0,
        "prob_branch": 0.03,
        "seed": 46, #
        "cores":n_cores
    }
    wandb.init(config=config, project="ising_model")
    run_branch()
    wandb.finish()


def run_sweep():
    
    pass 

if __name__ == "__main__":
    single_run()




