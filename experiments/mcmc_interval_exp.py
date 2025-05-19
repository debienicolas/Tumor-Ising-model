import os
import sys
import time

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
import networkx as nx


mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("branching_exp")

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BENCHMARK_DIR))

from branch_sim import MamSimulation
from utils.branch_sim_utils import plot_branch_network

output_dir = os.path.join("output", "branching_exp")
os.makedirs(output_dir, exist_ok=True)



# 3 major parameters:
# 1. prob_branch
# 2. fav -> f avoidance -> fs 
# 3. fchem ->  external field strength -> fc 

tmax = 150
f_prob_branch = np.arange(0.01, 0.1, 0.01)
f_fav = np.arange(-0.5, 0.0,0.1)
f_fchem = np.arange(0.1, 1.5, 0.1)

# f_prob_branch = [0.03]
# f_fav = [-0.1]
# f_fchem = [0.6]

# Function to calculate network statistics
def calculate_network_stats(G, coordinates, evolve):
    stats = {}
    
    # Basic network statistics
    stats["num_nodes"] = G.number_of_nodes()
    stats["num_edges"] = G.number_of_edges()
    stats["avg_degree"] = np.mean([d for _, d in G.degree()])
    
    # Advanced network statistics
    if nx.is_connected(G):
        stats["diameter"] = nx.diameter(G)
        stats["avg_path_length"] = nx.average_shortest_path_length(G)
    else:
        # Calculate stats for largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_subgraph = G.subgraph(largest_cc)
        stats["diameter_largest_cc"] = nx.diameter(largest_cc_subgraph)
        stats["avg_path_length_largest_cc"] = nx.average_shortest_path_length(largest_cc_subgraph)
        stats["largest_cc_size"] = len(largest_cc)
        stats["largest_cc_fraction"] = len(largest_cc) / G.number_of_nodes()
    
    # Unique branch count (from current_branch_id column)
    stats["num_branches"] = len(np.unique(coordinates[:, 3]))
    
    # Network extent
    stats["max_x"] = np.max(coordinates[:, 0])
    stats["min_x"] = np.min(coordinates[:, 0])
    stats["max_y"] = np.max(coordinates[:, 1])
    stats["min_y"] = np.min(coordinates[:, 1])
    stats["network_width"] = stats["max_x"] - stats["min_x"]
    stats["network_height"] = stats["max_y"] - stats["min_y"]
    stats["network_area"] = stats["network_width"] * stats["network_height"]
    
    # Growth metrics
    stats["node_density"] = stats["num_nodes"] / stats["network_area"] if stats["network_area"] > 0 else 0
    stats["branch_density"] = stats["num_branches"] / stats["network_area"] if stats["network_area"] > 0 else 0
    
    # Growth over time metrics
    nodes_per_timestep = evolve[1:tmax+1]  # Skip initial node
    stats["max_growth_rate"] = np.max(nodes_per_timestep)
    stats["avg_growth_rate"] = np.mean(nodes_per_timestep)
    stats["final_active_tips"] = evolve[-1]
    
    return stats


for f_fc in f_fchem:
    for f_pb in f_prob_branch:
        for f_fs in f_fav:

            # round the parameters to 2 decimal places
            f_fs = round(f_fs, 2)
            f_pb = round(f_pb, 2)
            f_fc = round(f_fc, 2)

            # start MLFlow run
            run_name = f"fav={f_fs}_pb={f_pb}_fc={f_fc}"
            with mlflow.start_run(run_name=run_name):
                # Log parameters
                mlflow.log_param("fav", f_fs)
                mlflow.log_param("prob_branch", f_pb)
                mlflow.log_param("fchem", f_fc)
                mlflow.log_param("tmax", tmax)
                
                # Run simulation
                start_time = time.time()
                mam = MamSimulation(tmax=tmax, prob_branch=f_pb, fav=f_fs, fchem=f_fc, graph_output=True)
                coordinates, evolve, G = mam.simulate()
                simulation_time = time.time() - start_time
                
                # Log simulation time
                mlflow.log_metric("simulation_time_seconds", simulation_time)
                
                # Calculate and log network statistics
                stats = calculate_network_stats(G, coordinates, evolve)
                for stat_name, stat_value in stats.items():
                    mlflow.log_metric(stat_name, stat_value)
                
                # Create and save network visualization
                title = f"fav={f_fs}, prob_branch={f_pb}, fchem={f_fc}"
                img_path = os.path.join(output_dir, f"fav{f_fs}_prob_branch{f_pb}_fchem{f_fc}.png")
                plot_branch_network(coordinates, evolve, title=title, save_path=img_path)
                
                # Log the network image to MLflow
                mlflow.log_artifact(img_path)
                
                # Create and log growth over time plot
                plt.figure(figsize=(10, 6))
                time_steps = np.arange(1, len(evolve))
                node_counts = [np.sum(evolve[:t]) for t in time_steps]
                plt.plot(time_steps, node_counts)
                plt.xlabel("Time Step")
                plt.ylabel("Total Node Count")
                plt.title(f"Growth Over Time\n{title}")
                plt.grid(True)
                growth_img_path = os.path.join(output_dir, f"growth_fav{f_fs}_pb{f_pb}_fc{f_fc}.png")
                plt.savefig(growth_img_path)
                plt.close()
                mlflow.log_artifact(growth_img_path)
                
                # Create and log network graph visualization
                plt.figure(figsize=(12, 12))
                pos = nx.get_node_attributes(G, 'pos')
                nx.draw(G, pos, node_size=10, with_labels=False)
                plt.title(f"Network Graph\n{title}")
                graph_img_path = os.path.join(output_dir, f"graph_fav{f_fs}_pb{f_pb}_fc{f_fc}.png")
                plt.savefig(graph_img_path)
                plt.close()
                mlflow.log_artifact(graph_img_path)

                # end MLFlow run
                mlflow.end_run()