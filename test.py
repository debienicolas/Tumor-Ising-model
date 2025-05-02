# test the time to get length of np array by using len or shape[0]

import numpy as np
import time

a = np.random.rand(1000000000)
a = a.astype(np.float16)
start = time.time()
print(len(a))
end = time.time()
print(f"Time taken: {end - start} seconds")

start = time.time()
print(a.shape[0])
end = time.time()
print(f"Time taken: {end - start} seconds")