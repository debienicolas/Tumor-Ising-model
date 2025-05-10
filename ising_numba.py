import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange

@njit
def initialize_grid(size):
    """Initialize a random spin grid."""
    return 2 * np.random.randint(0, 2, (size, size)) - 1

@njit
def calculate_energy_change(grid, i, j, J):
    """Calculate energy change when flipping the spin at position (i,j)."""
    size = grid.shape[0]
    spin = grid[i, j]
    
    # Get neighbors with periodic boundary conditions
    left = grid[i, (j - 1) % size]
    right = grid[i, (j + 1) % size]
    up = grid[(i - 1) % size, j]
    down = grid[(i + 1) % size, j]
    
    # Delta E = 2 * J * S_i * (sum of neighboring spins)
    return 2.0 * J * spin * (left + right + up + down)

@njit
def monte_carlo_step(grid, temperature, J):
    """Perform one Monte Carlo step (size*size attempted spin flips)."""
    size = grid.shape[0]
    
    for _ in range(size * size):
        # Choose a random site
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        
        # Calculate energy change if we flip this spin
        delta_E = calculate_energy_change(grid, i, j, J)
        
        # Metropolis acceptance criterion
        if delta_E <= 0 or np.random.random() < np.exp(-delta_E / temperature):
            grid[i, j] *= -1  # Flip the spin
    
    return grid

@njit
def calculate_energy(grid, J):
    """Calculate total energy of the system."""
    size = grid.shape[0]
    energy = 0.0
    
    for i in range(size):
        for j in range(size):
            spin = grid[i, j]
            
            # Consider only right and down neighbors to avoid double counting
            right = grid[i, (j + 1) % size]
            down = grid[(i + 1) % size, j]
            
            energy -= J * spin * (right + down)
    
    return energy

@njit
def calculate_magnetization(grid):
    """Calculate magnetization per site."""
    return np.sum(grid) / grid.size

def run_simulation(size=50, steps=500, temperature=2.27, J=1.0):
    """Run the Ising model simulation."""
    print("Running Ising model simulation...")
    print(f"Grid size: {size}x{size}")
    print(f"Temperature: {temperature}")
    
    # Initialize the grid
    grid = initialize_grid(size)
    
    # Start timing
    start_time = time.time()
    
    # Run the simulation
    for step in range(steps):
        grid = monte_carlo_step(grid, temperature, J)
        
        # Calculate and print observables every 100 steps
        if step % 100 == 0:
            energy = calculate_energy(grid, J)
            magnetization = calculate_magnetization(grid)
            print(f"Step: {step}, Energy: {energy}, Magnetization: {magnetization}")
    
    # End timing
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {elapsed_time:.4f} seconds")
    
    return grid

def visualize_grid(grid, filename="ising_visualization.png"):
    """Visualize and save the final grid state."""
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='coolwarm')
    plt.colorbar(label='Spin')
    plt.title('Ising Model Final Configuration')
    plt.savefig(filename)
    print(f"Visualization saved as '{filename}'")
    plt.close()

def save_grid(grid, filename="ising_final.dat"):
    """Save the grid to a file."""
    np.savetxt(filename, grid, fmt='%d')
    print(f"Final configuration saved to '{filename}'")

def main(size=50, steps=500, temperature=2.27, J=1.0):
    # Default parameters
    # size = 50
    # steps = 500
    # temperature = 2.27
    # J = 1.0
    
    # Run the simulation
    final_grid = run_simulation(size, steps, temperature, J)
    
    # Save and visualize results
    save_grid(final_grid)
    visualize_grid(final_grid)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    import sys
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
        if len(sys.argv) > 2:
            steps = int(sys.argv[2])
            if len(sys.argv) > 3:
                temperature = float(sys.argv[3])
        
        main(size, steps, temperature)
    else:
        main()