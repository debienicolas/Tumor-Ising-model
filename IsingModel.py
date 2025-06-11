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
    step_algorithm:str = "metropolis" # metropolis or glauber or wolff
    G: nx.Graph | None = None
    coords:np.ndarray | None = None
    evolve:np.ndarray | None = None
    start_time:int = 100
    save_frames:bool = True
    dim_cross:int = 1
    h:float = 0.0

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

    
    def get_config(self) -> dict:
        """
        Return the configuration of the model as a dictionary. 
        """
        return {
            "temp": self.temp,
            "J": self.J,
            "n_equilib_steps": self.n_equilib_steps,
            "n_mcmc_steps": self.n_mcmc_steps,
            "n_samples": self.n_samples,
            "n_sample_interval": self.n_sample_interval,
            "start_time": self.start_time,
            "save_frames": self.save_frames,
            "dim_cross": self.dim_cross,
            "step_algorithm": self.step_algorithm
        }


class IsingLatticeModel(IsingModel):
    """
    Ising model dataclass for lattice models.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes = np.ones(self.nodes.shape)
        self.neighbors = np.zeros((self.nodes.shape[0], 4))
        self.neighbors.fill(-1)
    

    def create_lattice(self, size:int):
        pass 
        