import numpy as np
import sys

# Create arrays with 1 million elements
m = 100
n = 1_000_000
data = np.random.choice([-1, 1], size=(n,m))

# Test different dtypes
arrays = {
    "float64 (default)": np.array(data),
    "int64": np.array(data, dtype=np.int64),
    "int32": np.array(data, dtype=np.int32),
    "int16": np.array(data, dtype=np.int16),
    "int8": np.array(data, dtype=np.int8),
    "bool": np.array(data == 1, dtype=bool)
}

# Calculate and print memory usage
print(f"Memory usage for {n:,} x {m:,} elements:")
for name, arr in arrays.items():
    memory_mb = arr.nbytes / (1024 * 1024)
    print(f"{name}: {memory_mb:.2f} MB")


# test the memory usage of 2D array and 1D array with same amount of elements
n = 1000
m = 1000
data = np.random.choice([-1, 1], size=(n,m))

# 2D array
memory_mb = data.nbytes / (1024 * 1024)
print(f"2D array: {memory_mb:.2f} MB")

# 1D array
data_1d = np.random.choice([-1, 1], size=n*m)
memory_mb = data_1d.nbytes / (1024 * 1024)
print(f"1D array: {memory_mb:.2f} MB")

