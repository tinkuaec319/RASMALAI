import numpy as np
import time

# Start time
start_time = time.time()

# Load the array
A = np.load('train_A.npy', mmap_mode=None)  # Load fully into memory

# Define the threshold
Q = 274887787
threshold = Q // 2

# Modify elements in A
A[A > threshold] -= Q

# End time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))

# Save the modified array
np.save('modified_train_A.npy', A)

