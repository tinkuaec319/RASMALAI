import os
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, OrthogonalMatchingPursuit
import pandas as pd
import time

# Start time
start_time = time.time()

# Parameters
start_hwt = 3
end_hwt = 40
Q = 41223389
half_value = Q // 2  # Integer division to get an integer

# Load data
secret_path = f'binary_secrets_h{start_hwt}_{end_hwt}/secret.npy'
train_b_path = f'binary_secrets_h{start_hwt}_{end_hwt}/train_b.npy'
modified_A_path = 'modified_train_A.npy'

secret_data = np.load(secret_path, allow_pickle=True)
train_b = np.load(train_b_path, allow_pickle=True)
modified_A = np.load(modified_A_path, mmap_mode='r')  # For large files

# NoMOD parameters
no_mod_percentage = 90
no_mod_fraction = no_mod_percentage / 100

# Number of samples to try
num_samples_list = [1000, 2000]

# List of models to try
models = [
    LinearRegression(),
    #ElasticNet(alpha=10.0, l1_ratio=0.5, max_iter=50000),
    OrthogonalMatchingPursuit()
]

# Recover secret function for binary secret
def recover_secret(model, filtered_A, filtered_b):
    model.fit(filtered_A, filtered_b)
    recovered_s = model.coef_ if hasattr(model, 'coef_') else model.feature_importances_
    # Threshold to convert to binary vector
    threshold = (np.max(recovered_s) + np.min(recovered_s)) / 2
    recovered_binary_s = (recovered_s > threshold).astype(int)
    return recovered_binary_s


# Create directory if it doesn't exist
output_dir = f"NoMod_{no_mod_percentage}%_results_for_h{start_hwt}_{end_hwt}_binary"
os.makedirs(output_dir, exist_ok=True)

# Loop over different sample sizes
for num_samples in num_samples_list:
    output_filename = os.path.join(output_dir, f"Secret_recovery_results_for_{num_samples}_samples_with_Nomod_{no_mod_percentage}%_with_model_names.txt")

    with open(output_filename, 'w') as output_file:
        b_column = 0
        true_s_column = 0

        for hamming_weight in range(start_hwt, end_hwt + 1):
            output_file.write(f"\n****************** Hamming Weight = {hamming_weight} ******************\n")

            for iteration in range(10):
                

                true_s = secret_data[:, true_s_column]
                b = train_b[:, b_column]

                modified_b = np.where(b > half_value, b - Q, b)
                result = np.dot(modified_A, true_s)
                abs_diff = np.abs(result - modified_b)

                no_mod_indices = np.where(abs_diff < half_value)[0]
                mod_indices = np.where(abs_diff >= half_value)[0]

                num_no_mod_samples = int(no_mod_fraction * num_samples)
                num_mod_samples = num_samples - num_no_mod_samples

                if len(mod_indices) < num_mod_samples:
                    output_file.write(f"Insufficient MOD data for hamming weight {hamming_weight}, iteration {iteration}, and {num_samples} samples\n")
                    continue

                if len(no_mod_indices) < num_no_mod_samples:
                    output_file.write(f"Insufficient NO MOD data for hamming weight {hamming_weight}, iteration {iteration}, and {num_samples} samples\n")
                    continue

                selected_no_mod_indices = np.random.choice(no_mod_indices, num_no_mod_samples, replace=False)
                selected_mod_indices = np.random.choice(mod_indices, num_mod_samples, replace=False)
                selected_indices = np.concatenate((selected_no_mod_indices, selected_mod_indices))
                np.random.shuffle(selected_indices)  # ✅ Shuffle to mix MOD and NoMOD

                filtered_A = modified_A[selected_indices]
                filtered_b = modified_b[selected_indices]

                successful_recoveries = 0
                successful_model_names = []

                model_name_map = {
                    'LinearRegression': 'LR',
                    # 'ElasticNet': 'Enet',
                    'OrthogonalMatchingPursuit': 'OMP'
                }

                iter_start_time = time.time()
                for model in models:
                    recovered_binary_s = recover_secret(model, filtered_A, filtered_b)
                    if np.array_equal(recovered_binary_s, true_s):
                        model_class_name = model.__class__.__name__
                        successful_model_names.append(model_name_map.get(model_class_name, model_class_name))
                        successful_recoveries += 1
                iter_end_time = time.time()
                elapsed = iter_end_time - iter_start_time
                
                true_s_column += 1
                b_column += 1

                # Write success count, successful models, and time
                model_str = ", ".join(successful_model_names)
                output_file.write(f"{successful_recoveries} \t ({model_str})\t\t\t{elapsed:.2f} seconds\n")

            output_file.write(f"******************Ending of hamming weight {hamming_weight} ******************\n\n\n\n")

# End time
end_time = time.time()
total_time = end_time - start_time
print(f"\n\nTotal time taken to execute the script: {total_time:.2f} seconds\n\n")
