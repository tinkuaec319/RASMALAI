import numpy as np
import pandas as pd
import time

# Start time
start_time = time.time()

# Load the data from train.prefix as if it were a CSV file with space as the delimiter
train_data = pd.read_csv('train.prefix', delimiter=' ', header=None)

# Load 256 dimensions of A
A = train_data.iloc[:, :256].values

# Define the threshold
Q = 842779
threshold = Q / 2

# Modify elements in A
A[A > threshold] -= Q

# End time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))

# Save the modified array to a file
np.save('modified_A.npy', A)
