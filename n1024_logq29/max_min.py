import numpy as np

# Memory-map the file for partial access
data = np.load('modified_train_A.npy', mmap_mode='r')

# Print basic info
print(f"Shape: {data.shape}, Dtype: {data.dtype}")

# Compute min and max safely
min_val = np.min(data)
max_val = np.max(data)

print(f"Min: {min_val}, Max: {max_val}")

