import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, OrthogonalMatchingPursuit
import pandas as pd
import time

# Start time
start_time = time.time()

# Parameters
start_hwt = 4
end_hwt = 9
Q = 41223389
half_value = Q // 2  # Integer division to get an integer

# Load data
secret_path = f'ternary_secrets_h{start_hwt}_{end_hwt}/secret.npy'
train_b_path = f'ternary_secrets_h{start_hwt}_{end_hwt}/train_b.npy'
modified_A_path = 'modified_train_A.npy'

secret_data = np.load(secret_path, allow_pickle=True)
train_b = np.load(train_b_path, allow_pickle=True)
modified_A = np.load(modified_A_path, mmap_mode='r')  # For large files

# NoMOD parameters
no_mod_percentage = 80
no_mod_fraction = no_mod_percentage / 100

# Number of samples to try
num_samples_list = [5000]

# List of models to try
models = [
    LinearRegression(),
    ElasticNet(alpha=10.0, l1_ratio=0.5, max_iter=50000),
    OrthogonalMatchingPursuit()
]

# Recover secret function for ternary secret
def recover_secret(model, filtered_A, filtered_b):
    model.fit(filtered_A, filtered_b)
    recovered_s = model.coef_ if hasattr(model, 'coef_') else model.feature_importances_

    # Compute the thresholds for ternary recovery
    min_value = np.min(recovered_s)
    mean_value = np.mean(recovered_s)
    max_value = np.max(recovered_s)

    threshold1 = (min_value + mean_value) / 2
    threshold2 = (max_value + mean_value) / 2

    # Convert the recovered values to ternary values
    recovered_ternary_s = np.where(recovered_s < threshold1, -1, np.where(recovered_s > threshold2, 1, 0))
    return recovered_ternary_s

# Loop over different sample sizes
for num_samples in num_samples_list:
    # Create the file name with the desired format
    output_filename = f"Secret_recovery_results_for_{num_samples}_samples_with_Nomod_{no_mod_percentage}%.txt"

    # Open the text file to write the results
    with open(output_filename, 'w') as output_file:
        b_column = 0
        true_s_column = 0

        for hamming_weight in range(start_hwt, start_hwt + 1):
            output_file.write(f"\n****************** Hamming Weight = {hamming_weight} ******************\n")
            no_mod_percentages = []

            # Write the table header
            header = f"{'H. Wt':<5} | {'Iteration':<9} | {'Model Name':<25} | {'#-1 true_s':<10} | {'#-1 MALAI_s':<11} | {'#1 true_s':<9} | {'#1 MALAI_s':<10} | {'Success/Fail':<13}"
            output_file.write(header + "\n")
            output_file.write("-" * len(header) + "\n")


            for iteration in range(10):
                # Extract columns
                true_s = secret_data[:, true_s_column]
                b = train_b[:, b_column]

                # Threshold b values
                modified_b = np.where(b > half_value, b - Q, b)

                # Compute dot product
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
                    recovered_ternary_s = recover_secret(model, filtered_A, filtered_b)

                    # Compare the number of -1's and 1's
                    true_negatives = np.sum(true_s == -1)
                    true_positives = np.sum(true_s == 1)

                    recovered_negatives = np.sum(recovered_ternary_s == -1)
                    recovered_positives = np.sum(recovered_ternary_s == 1)

                    # Check if the recovered secret matches the true secret
                    if np.array_equal(recovered_ternary_s, true_s):
                        success_status = "**SUCCESSFUL**"
                        successful_recoveries += 1
                    else:
                        success_status = "FAILED"

                    # Write to the file
                    output_file.write(f"{hamming_weight:<5} | {iteration+1:<9} | {model.__class__.__name__:<25} | {true_negatives:<10} | {recovered_negatives:<11} | {true_positives:<9} | {recovered_positives:<10} | {success_status:<13}\n")

                # End of iteration
            output_file.write(f"****************** End of Hamming Weight {hamming_weight} ******************\n\n\n\n")

# End time
end_time = time.time()
total_time = end_time - start_time
print(f"\n\nTotal time taken to execute the script: {total_time:.2f} seconds\n\n")
