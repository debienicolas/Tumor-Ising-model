import os
import sys
import time

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BENCHMARK_DIR))

from branch_sim import MamSimulation

output_dir = "bench_output"
os.makedirs(output_dir, exist_ok=True)

### benchmark the branch growth simulation -- original version

def branch_growth_benchmark():
    results = []

    t = np.arange(50,500,25)
    
    for tmax in tqdm(t):
        start_time = time.time()
        mam = MamSimulation(tmax=tmax)
        coordinates, evolve, G = mam.simulate()
        end_time = time.time()

        # get the number of nodes
        assert len(coordinates) == G.number_of_nodes()
        n_nodes = G.number_of_nodes()

        # get number of edges
        n_edges = G.number_of_edges()

        results.append({"tmax": tmax, "time": end_time - start_time, "n_nodes": n_nodes, "n_edges": n_edges})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "mod_1_no_njit_branching_bench.csv"), index=False)
    return df


if __name__ == "__main__":
    df = branch_growth_benchmark()








