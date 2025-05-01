"""
Utils for the branch simulation code. 

Main components:
- Coordinates object: np array with columns (x,y, parent_branch_id, current_branch_id)
- Evolve object: np array with the amount of coords added at each time step -> coords for t_i = coordinates[np.sum(evolve[:i])] has length of tmax + 1 as the last time 

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
import networkx as nx
from utils.gen_utils import graph_to_model_format, dimensional_crossover


def convert_branch_coords_to_graph(coordinates:np.ndarray, dist_thres:float=1.25, dim_cross:int=0) -> nx.Graph:
    """
    Convert the branch coordinates to a graph.

    Args:
        coordinates (np.ndarray): np array with columns (x,y, parent_branch_id, current_branch_id), there is no duplicate first node in the coordinates array
        distance_threshold (float): the distance threshold for the graph
    
    Returns:
        G: the graph with pos attribute for the x,y coordinates of the nodes
    """
    G = nx.Graph()
    for i in range(coordinates.shape[0]):
        G.add_node(i, pos=coordinates[i,0:2])
    for i in range(coordinates.shape[0]):
        for j in range(i+1,coordinates.shape[0]):
            # check if that they must have the same branch id or the parent branch id of i is the same as the current branch id of j
            if coordinates[i,3] == coordinates[j,3] or coordinates[i,3] == coordinates[j,2]:    
                if np.linalg.norm(coordinates[i,0:2] - coordinates[j,0:2]) < dist_thres:
                    G.add_edge(i, j)
    if dim_cross > 0:
        G = dimensional_crossover(G, dim_cross)
    return G

def model_format_at_time(coordinates:np.ndarray, evolve:np.ndarray, time_step:int, dim_cross:int=0) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the branch coordinates to model format at a given time step.
    First creates a graph from the coordinates, then converts it to nodes and neighbors.
    Uses the graph_to_model_format function which supports stacked graphs as well. 
    # TODO support growing stacked graphs 
    
    Args:
        coordinates (np.ndarray): np array with columns (x,y, parent_branch_id, current_branch_id)
        evolve (np.ndarray): np array with the amount of coords added at each time step -> coords for t_i = coordinates[np.sum(evolve[:i])] has length of tmax + 1 as the last time 
        time_step (int): the time step to convert
    
    Returns:
        nodes (np.ndarray): the nodes of the graph
        neighbors (np.ndarray): the neighbors of the nodes
    """
    time_coords = coordinates[:np.sum(evolve[:time_step])]
    G = convert_branch_coords_to_graph(time_coords, dim_cross=dim_cross)
    nodes, neighbors = graph_to_model_format(G)
    return nodes, neighbors

### Plotting functions ###

def branch_growth_animation(coordinates:np.ndarray, evolve:np.ndarray, output_path:str, start_time:int=5):
    """
    Creates an animation of the branch growth. The output is a .gif file. 

    Args:
        coordinates (np.ndarray): np array with columns (x,y, parent_branch_id, current_branch_id)
        evolve (np.ndarray): np array with the amount of coords added at each time step -> coords for t_i = coordinates[np.sum(evolve[:i])] has length of tmax + 1 as the last time 
        output_path (str): the path to save the animation
        start_time (int): the time step to start the animation
    """
    t_max = evolve.shape[0] - 1 # the last time step includes the active tips
    fig, ax = plt.subplots(figsize=(15, 15))
    ms = 1.5
    
    # Get all coordinates up to start_time
    idx = np.sum(evolve[:start_time])
    x, y = coordinates[:idx+1, 0], coordinates[:idx+1, 1]
    points = ax.plot(x, y, 'o', color='steelblue', markersize=ms)[0]
    start_point = ax.plot(coordinates[0, 0], coordinates[0, 1], 'x', color='firebrick', markersize=8)[0]
    
    # set the limits of the plot with respect to the coordinates of the final frame
    buffer = 20
    x_min, x_max = np.min(coordinates[:np.sum(evolve[:t_max]),0]), np.max(coordinates[:np.sum(evolve[:t_max]),0])
    y_min, y_max = np.min(coordinates[:np.sum(evolve[:t_max]),1]), np.max(coordinates[:np.sum(evolve[:t_max]),1])
    ax.set(xlim=(x_min-buffer, x_max+buffer), ylim=(y_min-buffer, y_max+buffer))
    
    def update(frame):
        idx = np.sum(evolve[:frame])
        x, y = coordinates[:idx+1, 0], coordinates[:idx+1, 1]
        points.set_data(x, y)
        return points, start_point
        
    ani = FuncAnimation(fig, update, frames=range(start_time, t_max), blit=True, interval=50)
    ani.save(output_path, writer='pillow')
    print(f"Branch growth animation saved to {output_path}")
    plt.close()

def plot_branch_graph(G:nx.Graph, save_path=None, show:bool=True, node_colors=None) -> Axes:
    """
    Plot the branch graph.

    Args:
        G (nx.Graph): the branch graph with pos attribute for the x,y coordinates of the nodes
        save_path (str): the path to save the plot
        show (bool): whether to show the plot
        node_colors (list): the colors of the nodes
    Returns:
        ax: the axis of the plot
    """
    fig, ax = plt.subplots(figsize=(15,15))
    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), with_labels=False, node_size=10, ax=ax, node_color=node_colors)
    if save_path:
        plt.savefig(save_path)
    elif show:
        plt.show()
    plt.close()
    return ax

def plot_branch_network(coordinates:np.ndarray, evolve:np.ndarray, end_time=None, show:bool=True, save_path=None) -> Axes:
    """
    Plots the branch coordinates.

    Args:
        coordinates (np.ndarray): np array with columns (x,y, parent_branch_id, current_branch_id)
        evolve (np.ndarray): np array with the amount of coords added at each time step -> coords for t_i = coordinates[np.sum(evolve[:i])] has length of tmax + 1 as the last time 
        end_time (int): the last time step to plot
        show (bool): whether to show the plot
        save_path (str): the path to save the plot
    
    Returns:
        ax: 
    """
    fig, ax = plt.subplots(figsize=(10,10))
    ms = 1.5

    if end_time is None:
        end_time = len(evolve)-1
    
    # The first coordinate is the root
    ax.plot(coordinates[0,0], coordinates[0,1], 'x', color='firebrick', markersize=8)
    # plot the regular points
    ax.plot(coordinates[:np.sum(evolve[:end_time]),0], coordinates[:np.sum(evolve[:end_time]),1], 'o', color='steelblue', markersize=ms)

    # if the end_time is None, plot the active tips with a different color
    if end_time is None:
        final_time_step = coordinates[np.sum(evolve[:end_time-1]):np.sum(evolve[:end_time])]
        ax.plot(final_time_step[:,0], final_time_step[:,1], 'o', color='C1', markersize=ms+1.5)

    max_x, min_x = np.max(coordinates[:,0]), np.min(coordinates[:,0])
    max_y, min_y = np.max(coordinates[:,1]), np.min(coordinates[:,1])
    ax.set_xlim(min_x-100,max_x+100)
    ax.set_ylim(min_y-100,max_y+100)
    
    plt.tick_params(    
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,   
        left=False,
        labelleft=False,
        labelbottom=False)
    if save_path:
        plt.savefig(save_path)
    elif show:
        plt.show()
    plt.close()
    return ax