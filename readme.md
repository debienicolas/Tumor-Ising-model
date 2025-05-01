# Tumor Ising model

This code consist of 3 main steps:  
1. Generating the topology
2. Running the Ising model
3. Analysis of the Ising model results (TO DO)

To do:
- fix the magnetization and energy calculations -> DONE
- Dimensional crossover 
- Add branching in time

- adding nodes, same as adjacent
- make sure the branching in time also supports dimensional crossover
- prerun some very large tmax simulations and save those coords and evolve arrays to file  
- calculate correlation statistics 

Pot. concerns:
- Metropolis single flip suffers from critical slow down? -> Wolff algorithm (cluster flip algorithms)


## Code

Scripts illustrating various functionalities are located in scripts folder:
- single_branch.py -> Run the ising model on a single branch at a certain time_step
- growing_branch.py -> Run the ising model on an in time growing branch structure

To Do:
- Fix the dimensional crossover and have dimensional crossover on growing branch

## optimizations
To be able to save every spin configuration, save them as np.int8 -> single byte per site.


## Experimentation

Questions to answer:
- 

## Results: Branching 

<div align="center">
<img src="output_mam_final/n_iter10000_J2.0_spins.png" alt="Ising Model Simulation" width="600">
</div>

<div align="center">
<img src="output_mam_final/magn_energy.png" alt="energy_magn" width="600">
</div>

<div align="center">
<img src="output_mam_final/animation_20250425_111009_T1.0_J2.0_iters10000.gif" alt="animation" width="600">
</div>


## Results: Grid 

<div align="center">
<img src="output_grid/ising_simulation_temps_spins_30x30_1.0_10000.png" alt="temp_grid" width="600">
</div>
<div align="center">
<img src="output_grid/ising_simulation_temps_30x30_1.0_10000.png" alt="temp_stats" width="600">
</div>


## Requirements
