import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import wandb
import time
import os
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go

from utils.gen_utils import create_lattice, graph_to_model_format, autocorr
from IsingModel import IsingModel
from main_tree import simulate_ising_model, simulate_ising_full
from branch_sim import MamSimulation
from utils.gen_utils import graph_to_model_format, dimensional_crossover
from utils.branch_sim_utils import plot_branch_network
from utils.gen_utils import create_lattice, create_3D_lattice
from utils.gen_utils import create_d_ary_tree

def get_branch_stats(G, coordinates):
    stats = {}
    # number of nodes and number of edges
    stats["num_nodes"] = G.number_of_nodes()
    stats["num_edges"] = G.number_of_edges()
    # Average degree of the graph
    stats["avg_degree"] = np.mean([d for _, d in G.degree()])
    
    # unique branch count (based on the current_branch_id column in the coordinates array)
    stats["num_branches"] = len(np.unique(coordinates[:, 3]))
    
    # calculate the average branch size (amount of nodes in the branch)
    branch_sizes = np.unique(coordinates[:, 3], return_counts=True)
    stats["avg_branch_size"] = np.mean(branch_sizes[1])
        
    
    return stats

def create_vis(G):
    # extract the positions from the node attrubutes pos
    pos = {node: G.nodes[node]["pos"] for node in G.nodes()}

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        line_width=2
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout())

    return fig

def run_branch():
    start_time_full = time.time()
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
    
    # log the visualization of the branch 
    wandb.log({'graph': wandb.Plotly(create_vis(G))})
    
    # log the branch stats 
    branch_stats = get_branch_stats(G, coords)
    wandb.log(branch_stats)

    # log the branch generation time
    wandb.log({
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
    
    
    # calculate the critical temperature for both the magnetization and energy smoothened curves
    # Find the temperature where the derivatives are maximum (the peaks in your plots)
    critical_temp_mag = temps[np.argmax(np.abs(dmag_dt))]
    critical_temp_energy = temps[np.argmax(denergy_dt)]
    
    wandb.log({
        "tc_mag": critical_temp_mag,
        "tc_energy": critical_temp_energy
    })
    
    
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
    total_time = end_time - start_time_full
    print(f"Simulation time: {sim_time} seconds")
    wandb.log({
        "sim_time": sim_time,
        "total_time": total_time
    })  

def run_dim_cross():
    wandb.init( project="ising_model")
    start_time_full = time.time()
    config = wandb.config
    
    # fix the issue with using numpy arrays in wandb
    temps = np.linspace(config["temps_min"], config["temps_max"], config["temps_num"]).tolist()

    
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
    
    ### Stack the branch n times
    if config["n_layers"] > 1:
        print(f"Stacking {config['n_layers']} layers")
        stacked_G = dimensional_crossover(G, config["n_layers"])
        print(f"Stacked graph has {stacked_G.number_of_nodes()} nodes and {stacked_G.number_of_edges()} edges")
    else:
        stacked_G = G
    
    nodes, neighbors = graph_to_model_format(stacked_G)
    end_time = time.time()
    branch_gen_time = end_time - start_time
    print(f"Branch generation time: {branch_gen_time} seconds")
    
    # log the visualization of the branch 
    wandb.log({'graph': wandb.Plotly(create_vis(stacked_G))})
    
    # log the branch stats 
    branch_stats = get_branch_stats(stacked_G, coords)
    wandb.log(branch_stats)

    # log the branch generation time
    wandb.log({
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
    
    
    # calculate the critical temperature for both the magnetization and energy smoothened curves
    # Find the temperature where the derivatives are maximum (the peaks in your plots)
    critical_temp_mag = temps[np.argmax(np.abs(dmag_dt))]
    critical_temp_energy = temps[np.argmax(denergy_dt)]
    
    wandb.log({
        "tc_mag": critical_temp_mag,
        "tc_energy": critical_temp_energy
    })
    
    
    # log the smoothed curves and derivatives
    for i in range(len(temps)):
        wandb.log({
            f"sm_magnetization": smooth_mag[i],
            f"sm_energy": smooth_energy[i],
            f"dm_dt": dmag_dt[i],
            f"de_dt": denergy_dt[i],
            f"temp": temps[i]
        })

    end_time = time.time()

    sim_time = end_time - start_time
    total_time = end_time - start_time_full
    print(f"Simulation time: {sim_time} seconds")
    wandb.log({
        "sim_time": sim_time,
        "total_time": total_time
    })

def run_lattice():
    wandb.init( project="ising_model")
    start_time_full = time.time()
    config = wandb.config
    
    # fix the issue with using numpy arrays in wandb
    temps = np.linspace(config["temps_min"], config["temps_max"], config["temps_num"]).tolist()

    
    # fix the issue with cores in wandb
    if config.get("cores") == 0:
        n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        print(f"Using {n_cores} cores from slurm environment")
    else:
        n_cores = config["cores"]

    # generate the branch
    start_time = time.time()
    if config["dims"] == 2:
        spins, neighbors = create_lattice(config["size"])
    elif config["dims"] == 3:
        spins, neighbors = create_3D_lattice(config["size"])
    else:
        raise ValueError(f"Invalid dimension: {config['dims']}")
    
    
    end_time = time.time()
    lattice_gen_time = end_time - start_time
    print(f"Lattice generation time: {lattice_gen_time} seconds")
    
    # log the branch generation time
    wandb.log({
        "lattice_gen_time": lattice_gen_time
    })

    # run the simulation
    start_time = time.time()
    spins = np.random.choice([-1, 1], size=spins.size)
    models = simulate_ising_full(
        nodes=spins,
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
    
    
    # calculate the critical temperature for both the magnetization and energy smoothened curves
    # Find the temperature where the derivatives are maximum (the peaks in your plots)
    critical_temp_mag = temps[np.argmax(np.abs(dmag_dt))]
    critical_temp_energy = temps[np.argmax(denergy_dt)]
    
    wandb.log({
        "tc_mag": critical_temp_mag,
        "tc_energy": critical_temp_energy
    })
    
    
    # log the smoothed curves and derivatives
    for i in range(len(temps)):
        wandb.log({
            f"sm_magnetization": smooth_mag[i],
            f"sm_energy": smooth_energy[i],
            f"dm_dt": dmag_dt[i],
            f"de_dt": denergy_dt[i],
            f"temp": temps[i]
        })

    end_time = time.time()

    sim_time = end_time - start_time
    total_time = end_time - start_time_full
    print(f"Simulation time: {sim_time} seconds")
    wandb.log({
        "sim_time": sim_time,
        "total_time": total_time
    })

def to_graph(spins, neighbors):
    G = nx.Graph()
    for i in range(spins.size):
        G.add_node(i)
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            if neighbors[i,j] != -1:
                G.add_edge(i, neighbors[i,j])
    return G

def get_tree_stats(G):
    stats = {}
    stats["num_nodes"] = G.number_of_nodes()
    stats["num_edges"] = G.number_of_edges()
    stats["avg_degree"] = np.mean([d for _, d in G.degree()])
    return stats

def create_vis(G):
    # use a tree layout
    pos = nx.drawing.layout.kamada_kawai_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        line_width=2
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout())

    return fig

def run_tree():
    wandb.init( project="ising_model")
    start_time_full = time.time()
    config = wandb.config
    
    # fix the issue with using numpy arrays in wandb
    temps = np.linspace(config["temps_min"], config["temps_max"], config["temps_num"]).tolist()

    
    # fix the issue with cores in wandb
    if config.get("cores") == 0:
        n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        print(f"Using {n_cores} cores from slurm environment")
    else:
        n_cores = config["cores"]

    # generate the branch
    start_time = time.time()
    spins, neighbors = create_d_ary_tree(depth=config["depth"], d=config["d"])
    
    # convert this model structure to a graph
    G = to_graph(spins, neighbors)
    
    # log the visualization of the branch 
    wandb.log({'graph': wandb.Plotly(create_vis(G))})
    
    # log the branch stats 
    branch_stats = get_tree_stats(G)
    wandb.log(branch_stats)
    
    
    end_time = time.time()
    lattice_gen_time = end_time - start_time
    print(f"Tree generation time: {lattice_gen_time} seconds")
    
    # log the branch generation time
    wandb.log({
        "tree_gen_time": lattice_gen_time
    })

    # run the simulation
    start_time = time.time()
    spins = np.random.choice([-1, 1], size=spins.size)
    models = simulate_ising_full(
        nodes=spins,
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
    
    
    # calculate the critical temperature for both the magnetization and energy smoothened curves
    # Find the temperature where the derivatives are maximum (the peaks in your plots)
    critical_temp_mag = temps[np.argmax(np.abs(dmag_dt))]
    critical_temp_energy = temps[np.argmax(denergy_dt)]
    
    wandb.log({
        "tc_mag": critical_temp_mag,
        "tc_energy": critical_temp_energy
    })
    
    
    # log the smoothed curves and derivatives
    for i in range(len(temps)):
        wandb.log({
            f"sm_magnetization": smooth_mag[i],
            f"sm_energy": smooth_energy[i],
            f"dm_dt": dmag_dt[i],
            f"de_dt": denergy_dt[i],
            f"temp": temps[i]
        })

    end_time = time.time()

    sim_time = end_time - start_time
    total_time = end_time - start_time_full
    print(f"Simulation time: {sim_time} seconds")
    wandb.log({
        "sim_time": sim_time,
        "total_time": total_time
    })
    
    

if __name__ == "__main__":
    run_tree()