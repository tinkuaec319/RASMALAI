import numpy as np

#Nearest prime number : 607817174357083

#Max(train_A.npy) : 607817174357048

#Next number of Max(train_A.npy) : 607817174438671

# Memory-map the file for partial access
data = np.load('modified_train_A.npy', mmap_mode='r')

# Print basic info
print(f"Shape: {data.shape}, Dtype: {data.dtype}")

# Compute min and max safely
min_val = np.min(data)
max_val = np.max(data)

print(f"Min: {min_val}, Max: {max_val}")

