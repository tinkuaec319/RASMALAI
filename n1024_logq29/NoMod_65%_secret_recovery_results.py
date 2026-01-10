import os
import numpy as np
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit
import time

start_time = time.time()

# ================= PARAMETERS =================
start_hwt = 3
end_hwt = 40
nomod_percentage = 0.65
Q = 274887787
half_value = Q // 2
num_iterations = 10
# ==============================================

# Initialize per-secret recovery tracker (380 entries)
secret_recovery_status = {
    (hwt, iter): False 
    for hwt in range(start_hwt, end_hwt + 1)
    for iter in range(num_iterations)
}

# Load data
secret_data = np.load(f'ternary_secrets_h{start_hwt}_{end_hwt}/secret.npy', allow_pickle=True)
train_b = np.load(f'ternary_secrets_h{start_hwt}_{end_hwt}/train_b.npy', allow_pickle=True)
modified_A = np.load('modified_train_A.npy', mmap_mode='r')

# Models and sample sizes
models = [LinearRegression(), OrthogonalMatchingPursuit()]
num_samples_list = [1000000, 2000000, 2500000]

# Recover secret function for ternary secret
def recover_secret(model, filtered_A, filtered_b):
    model.fit(filtered_A, filtered_b)
    recovered_s = model.coef_
    
    # Your original threshold calculation
    mean_val = np.mean(recovered_s)
    threshold_neg = (np.min(recovered_s) + mean_val) / 2
    threshold_pos = (np.max(recovered_s) + mean_val) / 2
    
    return np.select([recovered_s < threshold_neg, recovered_s > threshold_pos], [-1, 1], 0)

# Main processing loop
for num_samples in num_samples_list:
    output_dir = f"NoMod_{nomod_percentage*100:.0f}%_results_for_h{start_hwt}_{end_hwt}_ternary"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(
        output_dir, 
        f"Secret_recovery_results_for_{num_samples}_samples_with_Nomod_{nomod_percentage*100:.0f}%_ternary.txt"
    )
    
    with open(output_filename, 'w') as output_file:
        b_col, true_s_col = 0, 0
        
        for hamming_weight in range(start_hwt, end_hwt + 1):
            output_file.write(f"\n****************** Hamming Weight = {hamming_weight} ******************\n")
            
            for iteration in range(num_iterations):
                current_secret = (hamming_weight, iteration)
                
                # Skip already recovered secrets
                if secret_recovery_status[current_secret]:
                    output_file.write("2\t\t\tpreviously recovered\n")
                    b_col += 1
                    true_s_col += 1
                    continue
                
                # Load secret and corresponding b vector
                true_s = secret_data[:, true_s_col]
                b = train_b[:, b_col]
                modified_b = np.where(b > half_value, b - Q, b)
                
                # Calculate indices
                abs_diff = np.abs(np.dot(modified_A, true_s) - modified_b)
                mask = abs_diff < half_value
                no_mod_indices = np.where(mask)[0]
                mod_indices = np.where(~mask)[0]
                
                # Sample selection
                num_no_mod = int(nomod_percentage * num_samples)
                num_mod = num_samples - num_no_mod
                
                if len(no_mod_indices) < num_no_mod or len(mod_indices) < num_mod:
                    output_file.write("Insufficient data\n")
                    b_col += 1
                    true_s_col += 1
                    continue
                
                # Create dataset
                selected_indices = np.concatenate([
                    np.random.choice(no_mod_indices, num_no_mod, False),
                    np.random.choice(mod_indices, num_mod, False)
                ])
                np.random.shuffle(selected_indices)
                
                # Model processing
                iter_success = 0
                start_iter = time.time()
                
                for model in models:
                    recovered = recover_secret(model, 
                                             modified_A[selected_indices], 
                                             modified_b[selected_indices])
                    if np.array_equal(recovered, true_s):
                        iter_success += 1
                
                # Update recovery status if any model succeeded
                if iter_success >= 1:
                    secret_recovery_status[current_secret] = True
                
                # Write results
                elapsed = time.time() - start_iter
                output_file.write(f"{iter_success}\t\t\t{elapsed:.2f} seconds\n")
                
                b_col += 1
                true_s_col += 1

            output_file.write(f"******************Ending of hamming weight {hamming_weight} ******************\n\n\n\n")

print("Optimized processing complete")
print(f"\n\nTotal execution time: {time.time() - start_time:.2f} seconds\n\n")
