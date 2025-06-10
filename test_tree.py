import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class BinaryTreeIsingModel:
    """
    Ising Model simulation on a binary tree using Metropolis Monte Carlo.
    Based on the methodology from Mendoza (2014).
    """
    
    def __init__(self, depth=10, J=1.0, h=0.0):
        """
        Initialize binary tree Ising model.
        
        Args:
            depth: Tree depth (number of levels)
            J: Coupling strength
            h: External magnetic field strength
        """
        self.depth = depth
        self.J = J
        self.h = h  # External field
        self.nodes = []
        self.adjacency = defaultdict(list)
        self.spins = {}
        
        self._build_tree()
        self._initialize_spins()
    
    def _build_tree(self):
        """Build binary tree structure."""
        # Create nodes level by level
        for level in range(self.depth):
            level_nodes = []
            for i in range(2**level):
                node_id = sum(2**k for k in range(level)) + i
                level_nodes.append(node_id)
                
                # Connect to parent (except root)
                if level > 0:
                    parent_id = sum(2**k for k in range(level-1)) + i//2
                    self.adjacency[node_id].append(parent_id)
                    self.adjacency[parent_id].append(node_id)
            
            self.nodes.extend(level_nodes)
    
    def _initialize_spins(self):
        """Initialize spins randomly."""
        for node in self.nodes:
            self.spins[node] = random.choice([-1, 1])
    
    def get_energy(self):
        """Calculate total energy of the system."""
        # Bond energy: -J * ∑_{<i,j>} σ_i * σ_j
        bond_energy = 0.0
        visited_bonds = set()
        
        for node in self.nodes:
            for neighbor in self.adjacency[node]:
                bond = tuple(sorted([node, neighbor]))
                if bond not in visited_bonds:
                    bond_energy -= self.J * self.spins[node] * self.spins[neighbor]
                    visited_bonds.add(bond)
        
        # Field energy: -h * ∑_i σ_i
        field_energy = -self.h * sum(self.spins.values())
        
        return bond_energy + field_energy
    
    def get_local_energy_change(self, node):
        """Calculate energy change if node spin is flipped."""
        current_spin = self.spins[node]
        
        # Bond energy change: ΔE_bonds = 2J * σ_i * ∑_neighbors σ_j
        delta_E_bonds = 0.0
        for neighbor in self.adjacency[node]:
            delta_E_bonds += 2 * self.J * current_spin * self.spins[neighbor]
        
        # Field energy change: ΔE_field = 2h * σ_i
        delta_E_field = 2 * self.h * current_spin
        
        return delta_E_bonds + delta_E_field
    
    def get_magnetization(self):
        """Calculate net magnetization."""
        return sum(self.spins.values()) / len(self.nodes)
    
    def metropolis_step(self, beta):
        """Perform one Metropolis Monte Carlo step."""
        # Choose random node
        node = random.choice(self.nodes)
        
        # Calculate energy change
        delta_E = self.get_local_energy_change(node)
        
        # Metropolis acceptance criterion
        if delta_E <= 0 or random.random() < np.exp(-beta * delta_E):
            self.spins[node] *= -1  # Flip spin
            return True
        return False
    
    def monte_carlo_sweep(self, beta):
        """Perform one Monte Carlo sweep (N attempted flips)."""
        accepted = 0
        for _ in range(len(self.nodes)):
            if self.metropolis_step(beta):
                accepted += 1
        return accepted / len(self.nodes)
    
    def simulate_temperature(self, beta, n_equilibration=1000, n_measurement=1000):
        """
        Simulate at fixed temperature.
        
        Args:
            beta: Inverse temperature (1/kT, with J=1)
            n_equilibration: Sweeps for equilibration
            n_measurement: Sweeps for measurement
        
        Returns:
            tuple: (mean_magnetization, magnetization_history)
        """
        # Equilibration
        for _ in range(n_equilibration):
            self.monte_carlo_sweep(beta)
        
        # Measurement
        magnetizations = []
        for _ in range(n_measurement):
            self.monte_carlo_sweep(beta)
            magnetizations.append(abs(self.get_magnetization()))
        
        return np.mean(magnetizations), magnetizations
    
    def temperature_scan(self, T_min=0.1, T_max=3.0, n_temps=30, 
                        n_equilibration=1000, n_measurement=1000):
        """
        Scan over temperatures to find critical behavior.
        
        Returns:
            tuple: (temperatures, mean_magnetizations, all_magnetizations)
        """
        temperatures = np.linspace(T_min, T_max, n_temps)
        mean_mags = []
        all_mags = []
        
        print(f"Scanning {n_temps} temperatures from T={T_min} to T={T_max}")
        print(f"Tree depth: {self.depth}, Total nodes: {len(self.nodes)}")
        
        for i, T in enumerate(temperatures):
            beta = self.J / T  # beta = J/kT, with k=1
            mean_mag, mag_history = self.simulate_temperature(beta, n_equilibration, n_measurement)
            mean_mags.append(mean_mag)
            all_mags.append(mag_history)
            
            if i % 5 == 0:
                print(f"T = {T:.2f}, <|M|> = {mean_mag:.3f}")
        
        return temperatures, mean_mags, all_mags


def plot_results(temperatures, magnetizations):
    """Plot magnetization vs temperature."""
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, magnetizations, 'bo-', markersize=4)
    plt.xlabel('Temperature (T/J)')
    plt.ylabel('Mean |Magnetization|')
    plt.title('Ising Model on Binary Tree: Magnetization vs Temperature')
    plt.grid(True, alpha=0.3)
    
    # Find approximate critical temperature (steepest drop)
    dmag_dT = np.gradient(magnetizations, temperatures)
    critical_idx = np.argmin(dmag_dT)
    T_c = temperatures[critical_idx]
    plt.axvline(T_c, color='red', linestyle='--', alpha=0.7, 
                label=f'T_c ≈ {T_c:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"\nEstimated critical temperature: T_c ≈ {T_c:.2f}")
    print(f"Theoretical infinite tree value: T_c = 1.85")
    return T_c


def plot_magnetization_evolution(model, T=0.5, n_steps=10000):
    """Plot magnetization evolution during simulation."""
    beta = model.J / T
    magnetizations = []
    
    for step in range(n_steps):
        model.monte_carlo_sweep(beta)
        if step % 10 == 0:  # Sample every 10 sweeps
            magnetizations.append(model.get_magnetization())
    
    plt.figure(figsize=(10, 4))
    plt.plot(magnetizations)
    plt.xlabel('Monte Carlo Steps (×10)')
    plt.ylabel('Magnetization')
    plt.title(f'Magnetization Evolution at T/J = {T}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_with_without_field(depth=8, T=0.5, h_values=[0.0, 0.1, 0.5]):
    """Compare magnetization behavior with different field strengths."""
    plt.figure(figsize=(12, 8))
    
    for i, h in enumerate(h_values):
        model = BinaryTreeIsingModel(depth=depth, J=1.0, h=h)
        beta = 1.0 / T
        
        # Run simulation
        magnetizations = []
        for step in range(2000):
            model.monte_carlo_sweep(beta)
            if step % 10 == 0:
                magnetizations.append(model.get_magnetization())
        
        plt.subplot(len(h_values), 1, i+1)
        plt.plot(magnetizations, label=f'h = {h}', linewidth=1.5)
        plt.ylabel('Magnetization')
        if h == 1.0:
            plt.title(f'T/J = {T}, h = {h} (Critical Point!)', fontweight='bold')
        else:
            plt.title(f'T/J = {T}, h = {h}')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.xlabel('Monte Carlo Steps (×10)')
    plt.tight_layout()
    plt.show()
    
    print("Effect of external field:")
    print("- h = 0: Large fluctuations due to low-cost excitations")
    print("- h ≈ 1: Near critical point - enhanced fluctuations")
    print("- h > 1: Field suppresses fluctuations, stable magnetization")
    print("- Critical field h_c = J = 1.0 (Bruinsma 1984)")



def field_strength_scan(depth=8, T=0.5, h_max=1.0, n_fields=10):
    """Scan over field strengths to see stabilization effect."""
    h_values = np.linspace(0, h_max, n_fields)
    mag_means = []
    mag_stds = []
    
    for h in h_values:
        model = BinaryTreeIsingModel(depth=depth, J=1.0, h=h)
        beta = 1.0 / T
        
        # Equilibrate
        for _ in range(500):
            model.monte_carlo_sweep(beta)
        
        # Measure
        magnetizations = []
        for _ in range(1000):
            model.monte_carlo_sweep(beta)
            magnetizations.append(abs(model.get_magnetization()))
        
        mag_means.append(np.mean(magnetizations))
        mag_stds.append(np.std(magnetizations))
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(h_values, mag_means, yerr=mag_stds, fmt='o-', capsize=5)
    plt.xlabel('Field Strength (h)')
    plt.ylabel('Mean |Magnetization|')
    plt.title(f'Field Effect on Magnetization (T/J = {T})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return h_values, mag_means, mag_stds


def critical_field_test(depth=8, T=0.1, h_center=1.0, h_range=0.4, n_points=20):
    """
    Test critical behavior around theoretical h_c = J = 1.0 (Bruinsma 1984).
    Measures susceptibility and magnetization near the critical point.
    """
    print(f"=== Testing Critical Field Behavior ===")
    print(f"Theoretical prediction (Bruinsma 1984): h_c = J = 1.0")
    print(f"Testing range: h = {h_center-h_range/2:.2f} to {h_center+h_range/2:.2f}")
    print(f"Temperature: T = {T}")
    
    # Focus on region around h_c = 1.0
    h_values = np.linspace(h_center - h_range/2, h_center + h_range/2, n_points)
    
    magnetizations = []
    susceptibilities = []
    mag_fluctuations = []
    
    for h in h_values:
        model = BinaryTreeIsingModel(depth=depth, J=1.0, h=h)
        beta = 1.0 / T
        
        # Longer equilibration near critical point
        for _ in range(1000):
            model.monte_carlo_sweep(beta)
        
        # Measure magnetization and fluctuations
        mags = []
        for _ in range(2000):
            model.monte_carlo_sweep(beta)
            mags.append(model.get_magnetization())
        
        mags = np.array(mags)
        mean_mag = np.mean(np.abs(mags))
        mag_var = np.var(mags)
        
        # Susceptibility from fluctuation-dissipation theorem: χ = β⟨(M-⟨M⟩)²⟩
        susceptibility = beta * mag_var * len(model.nodes)  # per spin
        
        magnetizations.append(mean_mag)
        susceptibilities.append(susceptibility)
        mag_fluctuations.append(np.sqrt(mag_var))
        
        if abs(h - 1.0) < 0.05:  # Near theoretical h_c
            print(f"h = {h:.3f}: |M| = {mean_mag:.4f}, χ = {susceptibility:.2f}, σ_M = {np.sqrt(mag_var):.4f}")
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Magnetization
    ax1.plot(h_values, magnetizations, 'bo-', markersize=6, linewidth=2)
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Theory: h_c = J = 1.0')
    ax1.set_ylabel('Mean |Magnetization|')
    ax1.set_title('Critical Behavior Near h_c (Bruinsma 1984 Prediction)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Susceptibility
    ax2.plot(h_values, susceptibilities, 'go-', markersize=6, linewidth=2)
    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Theory: h_c = J = 1.0')
    ax2.set_ylabel('Susceptibility χ')
    ax2.set_title('Susceptibility Divergence at h_c')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Magnetization fluctuations
    ax3.plot(h_values, mag_fluctuations, 'mo-', markersize=6, linewidth=2)
    ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Theory: h_c = J = 1.0')
    ax3.set_xlabel('Field Strength h/J')
    ax3.set_ylabel('Magnetization Fluctuations σ_M')
    ax3.set_title('Fluctuations Near Critical Point')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Find empirical critical point from susceptibility maximum
    max_chi_idx = np.argmax(susceptibilities)
    h_c_empirical = h_values[max_chi_idx]
    
    print(f"\n=== Critical Point Analysis ===")
    print(f"Theoretical h_c = 1.000")
    print(f"Empirical h_c ≈ {h_c_empirical:.3f} (from χ maximum)")
    print(f"Deviation: {abs(h_c_empirical - 1.0):.3f}")
    print(f"Maximum susceptibility: χ_max = {np.max(susceptibilities):.1f}")
    
    return h_values, magnetizations, susceptibilities


def spin_flip_transitions_test(depth=8, T=0.05):
    """
    Test the series of spin-flip transitions predicted by Bruinsma (1984):
    h/J = 1 + 2/M for M = 1,2,3,...
    """
    print(f"=== Testing Spin-Flip Transitions ===")
    print(f"Bruinsma (1984) predicts transitions at h/J = 1 + 2/M")
    
    # Calculate theoretical transition points
    M_values = [1, 2, 3, 4, 5]
    h_transitions = [1 + 2/M for M in M_values]
    
    print("Theoretical transition points:")
    for M, h in zip(M_values, h_transitions):
        print(f"  M = {M}: h/J = {h:.3f}")
    
    # Test broader range including transitions
    h_values = np.linspace(0.2, 3.5, 40)
    
    magnetizations = []
    energies = []
    
    for h in h_values:
        model = BinaryTreeIsingModel(depth=depth, J=1.0, h=h)
        beta = 1.0 / T
        
        # Equilibrate
        for _ in range(800):
            model.monte_carlo_sweep(beta)
        
        # Measure
        mags = []
        engs = []
        for _ in range(1200):
            model.monte_carlo_sweep(beta)
            mags.append(abs(model.get_magnetization()))
            engs.append(model.get_energy() / len(model.nodes))
        
        magnetizations.append(np.mean(mags))
        energies.append(np.mean(engs))
    
    # Plot results with theoretical predictions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Magnetization
    ax1.plot(h_values, magnetizations, 'b-', linewidth=2, alpha=0.8)
    ax1.axvline(x=1.0, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Main: h_c = 1.0')
    
    # Mark predicted transitions
    colors = ['orange', 'green', 'purple', 'brown', 'pink']
    for i, (M, h_t) in enumerate(zip(M_values, h_transitions)):
        ax1.axvline(x=h_t, color=colors[i], linestyle='--', alpha=0.7, 
                   label=f'M={M}: h={h_t:.2f}')
    
    ax1.set_ylabel('Mean |Magnetization|')
    ax1.set_title('Spin-Flip Transitions in Random-Field Ising Model')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Energy
    ax2.plot(h_values, energies, 'g-', linewidth=2, alpha=0.8)
    ax2.axvline(x=1.0, color='red', linestyle='-', linewidth=2, alpha=0.8)
    
    for i, (M, h_t) in enumerate(zip(M_values, h_transitions)):
        ax2.axvline(x=h_t, color=colors[i], linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Field Strength h/J')
    ax2.set_ylabel('Energy per Spin')
    ax2.set_title('Energy vs Field Strength')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nNote: Transitions may be difficult to observe at finite T = {T}")
    print("For clearer transitions, try lower temperature (T → 0)")
    
    return h_values, magnetizations, energies


# Example usage
if __name__ == "__main__":
    # Create model (binary tree of depth 10)
    model = BinaryTreeIsingModel(depth=10, J=1.0, h=0.0)  # No field initially
    
    print("=== Random-Field Ising Model Validation ===")
    print(f"Based on Bruinsma (1984): Physical Review B 30, 289")
    print(f"Tree depth: {model.depth}")
    print(f"Total nodes: {len(model.nodes)}")
    print(f"Coupling J: {model.J}")
    print()
    
    # Example 1: Critical field test around h_c = 1.0
    print("1. Testing critical field behavior...")
    critical_field_test(depth=8, T=0.1, h_center=1.0, h_range=0.6, n_points=25)
    
    # Example 2: Spin-flip transitions
    print("\n2. Testing spin-flip transitions...")
    spin_flip_transitions_test(depth=8, T=0.05)
    
    # Example 3: Temperature dependence of critical field
    print("\n3. Temperature dependence...")
    T_values = [0.05, 0.1, 0.2, 0.3]
    h_c_values = []
    
    for T in T_values:
        print(f"\nTesting at T = {T}:")
        h_vals, mags, chis = critical_field_test(depth=8, T=T, h_center=1.0, h_range=0.4, n_points=15)
        h_c_empirical = h_vals[np.argmax(chis)]
        h_c_values.append(h_c_empirical)
        print(f"  Empirical h_c ≈ {h_c_empirical:.3f}")
    
    # Plot temperature dependence
    plt.figure(figsize=(10, 6))
    plt.plot(T_values, h_c_values, 'ro-', markersize=8, linewidth=2, label='Simulation')
    plt.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='Theory: h_c = 1.0')
    plt.xlabel('Temperature T/J')
    plt.ylabel('Critical Field h_c/J')
    plt.title('Temperature Dependence of Critical Field')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Example 4: Compare field effects at different temperatures
    print("\n4. Field effects at different temperatures...")
    compare_with_without_field(depth=8, T=0.1, h_values=[0.0, 0.5, 1.0, 1.5])
    
    print("\n=== Summary ===")
    print("Theoretical predictions (Bruinsma 1984):")
    print("• Main critical field: h_c = J = 1.0")
    print("• Spin-flip transitions: h/J = 1 + 2/M (M = 1,2,3,...)")
    print("  - h = 3.0 (M=1), h = 2.0 (M=2), h = 5/3 ≈ 1.67 (M=3), ...")
    print("• Critical behavior: χ ∝ 1/(h/J - 1) for h → 1⁺")
    print("• Ground state ferromagnetic for h < J")
    print("\nFinite-size and finite-temperature effects may cause deviations.")
    print("For exact agreement, take system size → ∞ and T → 0.")