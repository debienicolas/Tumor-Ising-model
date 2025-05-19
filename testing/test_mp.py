import sys, os
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from main_tree import simulate_ising_full, simulate_ising_model
from utils.gen_utils import create_lattice
from IsingModel import IsingModel

if __name__ == "__main__":
    nodes, neighbors = create_lattice(10)
    temps = np.linspace(0.01, 5, 100)

    n_equilib_steps = 10_000
    n_mcmc_steps = 1000
    n_samples = 100
    n_sample_interval = 10

    start_time = time.time()
    results = simulate_ising_full(
        nodes=nodes,
        neighbors=neighbors,
        J=1,
        n_equilib_steps=n_equilib_steps,
        n_mcmc_steps=n_mcmc_steps,
        n_samples=n_samples,
        n_sample_interval=n_sample_interval,
        temps=temps
    )
    end_time = time.time()
    print(f"With multiprocessing: Time taken: {end_time - start_time} seconds")

    # without multiprocessing
    start_time = time.time()
    results_no_mp = []
    for t in temps:
        model = IsingModel(
            nodes=nodes,
            neighbors=neighbors,
            J=1,
            temp=t,
            n_equilib_steps=n_equilib_steps,
            n_mcmc_steps=n_mcmc_steps,
            n_samples=n_samples,
            n_sample_interval=n_sample_interval
        )
        results_no_mp.append(simulate_ising_model(model))
    end_time = time.time()
    print(f"Without multiprocessing: Time taken: {end_time - start_time} seconds")

    