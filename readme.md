# Tumor Ising model

This code consist of 3 main steps:  
1. Generating the topology
2. Running the Ising model
3. Analysis of the Ising model results (TO DO)

## Results: Branching 

![Ising Model Simulation](output_mam_final/n_iter10000_J2.0_spins.png)

![energy_magn](output_mam_final/magn_energy.png)

![animation](output_mam_final/animation_20250425_111009_T1.0_J2.0_iters10000.gif)


## Results: Grid 

![temp_grid](output_grid/ising_simulation_temps_spins_30x30_1.0_10000.png)
![temp_stats](output_grid/ising_simulation_temps_30x30_1.0_10000.png)

## Background: The Ising model in physics

The Ising model was first introduced in the 1920s by Ernest Ising, which concerns itself with the physics of phase transitions. More specifically the model was originally used to obtain a better understanding of ferromagnetism and especially "spontaneous magnetization".

The first component of the Ising model is the topology structure. The classical starting point would be a 2D lattice with "wrap around" boundary conditions which turns it into a torus. Each lattice site is assigned an independent variable $\sigma_i$ for $i=1,...,N$.
The independent variables $\sigma_i$ can be in 2 possible states, $\sigma_i = \pm 1$ reflecting the physical assumption that only 2 possibilities exist such as spin up/down(Potts model generalizes to multiple states). Assigning each lattice point a state gives us a configuration fo the system $(\sigma_1,..., \sigma_N)$.    

A second component is called the *Hamiltonian* function which denotes the energy of a configuration $\sigma$. The Ising Hamiltoniain is formulated as follows:  
```math
H(\sigma) = -\sum_{\langle i,j\rangle} J_{i,j} \sigma_i \sigma_j - \mu \sum
```




1D results: each site only interacts with its left and right neighbor -> no phase transition.



## Requirements
