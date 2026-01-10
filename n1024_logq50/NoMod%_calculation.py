import numpy as np
import pandas as pd
import time

# Start time
start_time = time.time()

# Parameters
start_hwt = 3
end_hwt = 40
Q = 607817174438671
half_value = Q // 2

# Load data
secret_path = f'ternary_secrets_h{start_hwt}_{end_hwt}/secret.npy'
train_b_path = f'ternary_secrets_h{start_hwt}_{end_hwt}/train_b.npy'
modified_A_path = 'modified_train_A.npy'

secret_data = np.load(secret_path, allow_pickle=True)
train_b = np.load(train_b_path, allow_pickle=True)
modified_A = np.load(modified_A_path, mmap_mode='r')  # For large files

# Output file
output_filename = f'NoMod_percentages_for_each_hwt_{start_hwt}__to__hwt_{end_hwt}_ternary.txt'

with open(output_filename, 'w') as output_file:
    total_columns = (end_hwt - start_hwt + 1) * 10
    assert secret_data.shape[1] == total_columns
    assert train_b.shape[1] == total_columns

    b_column = 0
    true_s_column = 0

    for hamming_weight in range(start_hwt, end_hwt + 1):
        output_file.write(f"****************** Hamming Weight = {hamming_weight} ******************\n")
        no_mod_percentages = []

        for iteration in range(10):
            # Extract columns
            true_s = secret_data[:, true_s_column]
            b = train_b[:, b_column]

            # Threshold b values
            modified_b = np.where(b > half_value, b - Q, b)

            # Compute dot product
            result = np.dot(modified_A, true_s)
            abs_diff = np.abs(result - modified_b)

            # Count NoMOD values
            no_mod_indices = np.where(abs_diff < half_value)[0]
            no_mod_percentage = (len(no_mod_indices) / len(b)) * 100
            no_mod_percentages.append(no_mod_percentage)

            # Write iteration result
            output_file.write(f"Iteration {iteration + 1}: NoMOD % = {no_mod_percentage:.2f}%\n")

            # Move to next columns
            true_s_column += 1
            b_column += 1

        # Average over 10 iterations
        avg_no_mod = sum(no_mod_percentages) / 10
        output_file.write(f"Average NoMOD % for HW={hamming_weight}: {avg_no_mod:.2f}%\n\n\n\n")

print(f"\n✅ Done. Results saved in '{output_filename}'")



# End time
end_time = time.time()
total_time = end_time - start_time
print(f"\n\nTotal time taken to execute the script: {total_time:.2f} seconds\n\n")