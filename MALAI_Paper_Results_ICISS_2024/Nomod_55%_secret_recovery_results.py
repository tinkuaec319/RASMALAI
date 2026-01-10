import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, OrthogonalMatchingPursuit
import pandas as pd
import time

# Start time
start_time = time.time()

# Load the data
train_data = pd.read_csv('train.prefix', delimiter=' ', header=None)
modified_A = np.load('A_modified.npy')
secret_data = np.load('secret.npy')

# Constants
Q = 842779
half_value = Q / 2
no_mod_percentage = 55
no_mod_fraction = no_mod_percentage / 100

# Hamming weights from 3 to 40
hamming_weights = range(3, 41)

# Number of samples to try
num_samples_list = [100000, 200000, 500000, 750000, 1000000]

# List of models to try
models = [
    LinearRegression(),
    ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000), 
    OrthogonalMatchingPursuit()
]

# Recover secret function
def recover_secret(model, filtered_A, filtered_b):
    model.fit(filtered_A, filtered_b)
    recovered_s = model.coef_ if hasattr(model, 'coef_') else model.feature_importances_
    # Threshold to convert to binary vector
    threshold = (np.max(recovered_s) + np.min(recovered_s)) / 2
    recovered_binary_s = (recovered_s > threshold).astype(int)
    return recovered_binary_s

# Loop over different sample sizes
for num_samples in num_samples_list:
    # Create the file name with the desired format
    output_filename = f"Secret_recovery_results_for_{num_samples}_samples_with_Nomod_{no_mod_percentage}%.txt"

    # Open the text file to write the results
    with open(output_filename, 'w') as output_file:

        # Loop over Hamming weights
        for hamming_weight in hamming_weights:
            output_file.write(f"******************hamming weight = {hamming_weight} ******************\n")
            
            # Loop over iterations for true_s and b
            for iteration in range(10):
                true_s_column = (hamming_weight - 3) * 10 + iteration
                b_column = true_s_column + 257
                true_s = secret_data[:, true_s_column]
                b = train_data.iloc[:, b_column].values

                # Modify b based on condition
                modified_b = np.where(b > half_value, b - Q, b)

                # Calculate modified_A * true_s
                result = np.dot(modified_A, true_s)
                abs_diff = np.abs(result - modified_b)

                # Identify NoMOD and MOD indices
                no_mod_indices = np.where(abs_diff < half_value)[0]
                mod_indices = np.where(abs_diff >= half_value)[0]

                successful_recoveries = 0
                num_no_mod_samples = int(no_mod_fraction * num_samples)
                num_mod_samples = num_samples - num_no_mod_samples

                # Check if there's enough MOD data
                if len(mod_indices) < num_mod_samples:
                    output_file.write(f"Insufficient MOD data for hamming weight {hamming_weight}, iteration {iteration}, and {num_samples} samples\n")
                    continue

                # Select samples from NoMOD and MOD
                selected_no_mod_indices = np.random.choice(no_mod_indices, num_no_mod_samples, replace=False)
                selected_mod_indices = np.random.choice(mod_indices, num_mod_samples, replace=False)

                # Combine selected indices
                selected_indices = np.concatenate((selected_no_mod_indices, selected_mod_indices))

                # Filter A and b using the selected indices
                filtered_A = modified_A[selected_indices]
                filtered_b = modified_b[selected_indices]

                # Try each model
                for model in models:
                    recovered_binary_s = recover_secret(model, filtered_A, filtered_b)
                    
                    # Check if recovered secret matches true_s
                    if np.array_equal(recovered_binary_s, true_s):
                        successful_recoveries += 1

                # Write whether secret was recovered (1) or not (0)
                output_file.write(f"{successful_recoveries}\n")
            
            # End of hamming weight
            output_file.write(f"******************Ending of hamming weight {hamming_weight} ******************\n\n")

# End time
end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken to execute the script: {total_time:.2f} seconds")
