#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>

class IsingModel {
private:
    int size;                       // Grid size (NxN)
    std::vector<int> spins;         // Spin grid
    double temperature;             // Temperature in units where k_B = 1
    double J;                       // Coupling constant
    std::mt19937 rng;               // Random number generator
    std::uniform_real_distribution<double> dist;  // Uniform distribution [0,1]
    std::uniform_int_distribution<int> site_dist; // For site selection

public:
    IsingModel(int size, double temp, double J)
        : size(size), temperature(temp), J(J), 
          spins(size * size), 
          rng(std::random_device{}()),
          dist(0.0, 1.0),
          site_dist(0, size - 1)
    {
        // Initialize the grid randomly with +1 or -1 spins
        for (int i = 0; i < size * size; ++i) {
            spins[i] = (dist(rng) < 0.5) ? 1 : -1;
        }
    }

    // Calculate energy change when flipping the spin at (i,j)
    double calculateEnergyChange(int i, int j) {
        int idx = i * size + j;
        int spin = spins[idx];
        
        // Get neighbors with periodic boundary conditions
        int left = spins[i * size + ((j - 1 + size) % size)];
        int right = spins[i * size + ((j + 1) % size)];
        int up = spins[((i - 1 + size) % size) * size + j];
        int down = spins[((i + 1) % size) * size + j];
        
        // deltaE = 2 * J * s_i * sum(neighbors)
        return 2.0 * J * spin * (left + right + up + down);
    }

    // Perform one Monte Carlo step (size*size attempted spin flips)
    void monteCarloStep() {
        for (int n = 0; n < size * size; ++n) {
            // Choose a random site
            int i = site_dist(rng);
            int j = site_dist(rng);
            
            // Calculate energy change if we flip this spin
            double deltaE = calculateEnergyChange(i, j);
            
            // Metropolis acceptance criterion
            if (deltaE <= 0 || dist(rng) < exp(-deltaE / temperature)) {
                spins[i * size + j] *= -1;  // Flip the spin
            }
        }
    }

    // Calculate total energy of the system
    double calculateEnergy() {
        double energy = 0.0;
        
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                int idx = i * size + j;
                int spin = spins[idx];
                
                // Consider only right and down neighbors to avoid double counting
                int right = spins[i * size + ((j + 1) % size)];
                int down = spins[((i + 1) % size) * size + j];
                
                energy -= J * spin * (right + down);
            }
        }
        
        return energy;
    }

    // Calculate magnetization per site
    double calculateMagnetization() {
        int sum = 0;
        for (int i = 0; i < size * size; ++i) {
            sum += spins[i];
        }
        return static_cast<double>(sum) / (size * size);
    }

    // Save the grid to a file
    void saveGrid(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        
        // Save as a grid of +/-1
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                file << spins[i * size + j] << " ";
            }
            file << "\n";
        }
        
        file.close();
    }
};

int main(int argc, char* argv[]) {
    // Default parameters
    int size = 50;              // Grid size
    int steps = 500;            // Number of Monte Carlo steps
    double temperature = 2.27;  // Critical temperature for the 2D Ising model
    double J = 1.0;             // Coupling constant
    
    // Parse command line arguments if provided
    if (argc > 1) size = std::stoi(argv[1]);
    if (argc > 2) steps = std::stoi(argv[2]);
    if (argc > 3) temperature = std::stod(argv[3]);
    
    std::cout << "Running Ising model simulation..." << std::endl;
    std::cout << "Grid size: " << size << "x" << size << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    
    // Initialize the model
    IsingModel model(size, temperature, J);
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run the simulation
    for (int step = 0; step < steps; ++step) {
        model.monteCarloStep();
        
        // Calculate and print observables every 100 steps
        if (step % 100 == 0) {
            double energy = model.calculateEnergy();
            double magnetization = model.calculateMagnetization();
            std::cout << "Step: " << step 
                      << ", Energy: " << energy 
                      << ", Magnetization: " << magnetization << std::endl;
        }
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "Simulation completed in " << elapsed.count() << " seconds" << std::endl;
    
    // Save the final configuration
    model.saveGrid("ising_final.dat");
    std::cout << "Final configuration saved to 'ising_final.dat'" << std::endl;
    
    return 0;
}