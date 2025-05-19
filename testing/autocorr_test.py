"""
Use this script to test the autocorrelation function

Test this on a lattice first
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

#from main import IsingModel, simulate_ising_model
from IsingModel import IsingModel
from main_tree import simulate_ising_model
from branch_sim import MamSimulation
from utils.gen_utils import graph_to_model_format, autocorr

from utils.gen_utils import create_binary_tree



def exp_decay(t, tau):
    return np.exp(-t/tau)


# Define temperatures to analyze
temperatures = [1.5, 2.0, 2.269, 3.0, 4.0]  # 2.269 is the critical temperature for 2D Ising model
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Store tau values to plot vs temperature later
tau_values = []

# Create lattice
nodes, neighbors = create_binary_tree(depth=15)
print(nodes.shape)
#coordinates, evolve, G = MamSimulation(tmax=100, seed=43).simulate()
#nodes, neighbors = graph_to_model_format(G)




# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Loop through temperatures
for i, temp in enumerate(temperatures):
    # Simulate the model
    nodes = np.random.choice([-1,1], size=nodes.shape)
    model = IsingModel(nodes=nodes, neighbors=neighbors, temp=temp, J=1.0, 
                      n_equilib_steps=10_000, n_mcmc_steps=5000, 
                      n_sample_interval=1, n_samples=5000, 
                      step_algorithm="wolff")
    model = simulate_ising_model(model)
    x = model.energy_samples
    print("Model all energy length:", x.shape)
    # Calculate autocorrelation
    corr = autocorr(x)

    # use gaussian filter to smooth the data
    corr = gaussian_filter1d(corr, sigma=4)

    
    # Fit the data to an exponential decay function
    # Use only data where correlation is positive and above some small threshold
    # to avoid noise in the tail affecting the fit
    positive_mask = corr > 0.05
    if np.sum(positive_mask) > 5:  # Ensure we have enough points for fitting
        lags = np.arange(len(corr))
        max_lag = lags[-1]
        fit_lags = lags[positive_mask]
        fit_corr = corr[positive_mask]
        
        # Fit the data
        try:
            popt, pcov = curve_fit(exp_decay, fit_lags, fit_corr, p0=[50])
            tau_fit = popt[0]
            tau_values.append((temp, tau_fit))
            print(f"T = {temp}: tau = {tau_fit:.3f}")
            
            # Generate fitted curve
            fitted_corr = exp_decay(lags, tau_fit)
            
            # Plot data and fit on linear scale
            ax1.plot(lags, corr, '-', color=colors[i], 
                    label=f'T = {temp:.2f}, τ = {tau_fit:.1f}')
            ax1.plot(lags, fitted_corr, '--', color=colors[i], alpha=0.7)
            
            # Plot data and fit on semi-log scale
            ax2.semilogy(lags, np.maximum(corr, 1e-10), '-', color=colors[i])
            ax2.semilogy(lags, fitted_corr, '--', color=colors[i], alpha=0.7)
        except:
            print(f"Fitting failed for T = {temp}")
            ax1.plot(lags, corr, '-', color=colors[i], 
                    label=f'T = {temp:.2f} (fit failed)')
            ax2.semilogy(lags, np.maximum(corr, 1e-10), '-', color=colors[i])
    else:
        print(f"Not enough positive correlation points for T = {temp}")
        ax1.plot(lags, corr, '-', color=colors[i], 
                label=f'T = {temp:.2f} (insufficient data)')
        ax2.semilogy(lags, np.maximum(corr, 1e-10), '-', color=colors[i])

# Configure linear plot
ax1.set_xlabel('Lag t')
ax1.set_ylabel('C(t)')
ax1.set_title('Autocorrelation Functions (Linear Scale)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, max_lag)
ax1.set_ylim(-0.1, 1.1)

# Configure semi-log plot
ax2.set_xlabel('Lag t')
ax2.set_ylabel('C(t)')
ax2.set_title('Autocorrelation Functions (Log Scale)')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, max_lag)
ax2.set_ylim(1e-3, 1)
plt.tight_layout()
plt.show()

# Plot tau vs temperature if we have enough data points
if len(tau_values) > 1:
    temps, taus = zip(*tau_values)
    
    plt.figure(figsize=(8, 6))
    plt.plot(temps, taus, 'o-')
    plt.axvline(x=2.269, linestyle='--', color='gray', alpha=0.7, label='Critical temp (2.269)')
    plt.xlabel('Temperature')
    plt.ylabel('Correlation time τ')
    plt.title('Correlation Time vs Temperature')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()