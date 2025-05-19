import numpy as np

def autocorrelation_function(samples, max_lag=None):
    """
    Calculate the autocorrelation function c(k) for a list of samples.
    
    Parameters:
    -----------
    samples : array_like
        List or array of energy samples or other observable
    max_lag : int, optional
        Maximum lag to calculate. If None, uses N/4
        
    Returns:
    --------
    lags : ndarray
        Array of lag values from 0 to max_lag
    corr : ndarray
        Array of autocorrelation values corresponding to each lag
    """
    samples = np.array(samples)
    N = len(samples)
    
    if max_lag is None:
        # A common practice is to limit the maximum lag to N/4
        max_lag = N // 4
    
    # Initialize array to hold autocorrelation values
    corr = np.zeros(max_lag + 1)
    
    # Calculate autocorrelation for each lag k
    for k in range(max_lag + 1):
        # Select the valid range of indices for this lag
        indices = np.arange(0, N - k)
        
        # Get the samples and the lagged samples
        x_i = samples[indices]
        x_i_plus_k = samples[indices + k]
        
        # Calculate the mean of the samples in this window
        mean_x = np.mean(x_i)
        
        # Apply the autocorrelation formula from the equation
        numerator = np.sum(x_i_plus_k * (x_i - mean_x))
        denominator = N - k
        
        corr[k] = numerator / denominator
    
    # Normalize the autocorrelation function
    corr = corr / corr[0]
    
    lags = np.arange(max_lag + 1)
    return lags, corr

# Example usage:
if __name__ == "__main__":
    # Example data - replace with your energy samples
    energy_samples = np.random.normal(size=10000, scale=10)
    
    # Calculate the autocorrelation function
    lags, acf = autocorrelation_function(energy_samples, max_lag=50)
    
    # You can plot the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(lags, acf)
    plt.xlabel('Lag k')
    plt.ylabel('Autocorrelation c(k)')
    plt.title('Autocorrelation Function')
    plt.grid(True)
    #plt.semilogy()  # Use logarithmic scale on y-axis as in the paper
    plt.show()
    
    # Optionally, you can estimate the autocorrelation time tau
    # A simple estimate is the lag where c(k) drops below 1/e
    try:
        tau = lags[np.where(acf < 1/np.e)[0][0]]
        print(f"Estimated autocorrelation time: {tau}")
    except IndexError:
        print("Could not estimate autocorrelation time - increase max_lag")