from dataclasses import dataclass

import networkx as nx
import numpy as np

@dataclass
class IsingModel:
    """
    Ising model dataclass. 
    Holds input arguments, config parameters and is used to store simulation results.
    """

    nodes:np.ndarray
    neighbors:np.ndarray
    temp:float
    J:float
    n_equilib_steps:int
    n_mcmc_steps:int
    n_samples:int = 10
    n_sample_interval:int = 1
    G: nx.Graph | None = None
    coords:np.ndarray | None = None
    evolve:np.ndarray | None = None
    start_time:int = 100
    save_frames:bool = True
    dim_cross:int = 1

    @property
    def beta(self):
        return 1/self.temp       

    def save_results(self, spins:np.ndarray, magn:float, energy:float, specific_heat:float, susceptibility:float) -> None:
        """
        Save the results of the simulation to the dataclass.

        Args:
            spins: 2D stacked array of spins at each time step. Shape: (n_iterations_total, n_nodes)
            magn: The magnetization of the system: average over the amount of samples
            energy: The energy of the system: average over the amount of samples
        """

        # save the results
        self.spins = spins
        self.total_magn = magn
        self.total_energy = energy
        self.specific_heat = specific_heat
        self.susceptibility = susceptibility
        # set a flag to indicate the config has been simulated
        self.simulated = True

    
