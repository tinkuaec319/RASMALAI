import numpy as np
import time
import os

# ================= PARAMETERS =================
start_hwt = 3
end_hwt = 40
num_iterations = 10
Q = 41223389
half_value = Q // 2
thresholds = [0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10]
output_dir = "Threshold_Analysis_Results_for_ternary"
# ==============================================

# Load data
secret_data = np.load(f'ternary_secrets_h{start_hwt}_{end_hwt}/secret.npy', allow_pickle=True)
train_b = np.load(f'ternary_secrets_h{start_hwt}_{end_hwt}/train_b.npy', allow_pickle=True)
modified_A = np.load('modified_train_A.npy', mmap_mode='r')

os.makedirs(output_dir, exist_ok=True)

def process_threshold(threshold):
    """Process all HWTs for a single threshold"""
    filename = os.path.join(output_dir, f"Threshold_{threshold}Q_analysis_ternary.txt")
    
    with open(filename, 'w') as file:
        # Write header
        file.write("HammingWeight  OriginalSamples  FilteredSamples  OriginalNoMod%  FilteredNoMod%  Improvement%\n")
        file.write("-------------------------------------------------------------------------------------------\n")
        
        # Process each HWT
        for hwt in range(start_hwt, end_hwt + 1):
            hwt_data = {
                'original_samples': [],
                'filtered_samples': [],
                'original_nomod': [],
                'filtered_nomod': []
            }
            
            # Process 10 iterations
            for iteration in range(num_iterations):
                # Load data (adjust paths as needed)
                true_s = secret_data[:, (hwt-start_hwt)*num_iterations + iteration]
                b = train_b[:, (hwt-start_hwt)*num_iterations + iteration]
                modified_b = np.where(b > half_value, b - Q, b)
                
                # Calculate original NoMod
                abs_diff = np.abs(np.dot(modified_A, true_s) - modified_b)
                no_mod_mask = abs_diff < half_value
                original_nomod = np.mean(no_mod_mask) * 100
                
                # Vector transformation and filtering
                hamming_weight = np.count_nonzero(true_s)
                A_adjusted = modified_A - (modified_b/hamming_weight)[:, np.newaxis]
                vector_A = np.sum(A_adjusted, axis=1)
                filter_mask = (vector_A > -threshold*Q) & (vector_A < threshold*Q)
                
                # Collect metrics
                hwt_data['original_samples'].append(len(b))
                hwt_data['filtered_samples'].append(np.sum(filter_mask))
                hwt_data['original_nomod'].append(original_nomod)
                hwt_data['filtered_nomod'].append((np.sum(no_mod_mask & filter_mask) / np.sum(filter_mask)) * 100) 

            
            # Calculate averages
            avg_line = (
                hwt,
                np.mean(hwt_data['original_samples']),
                np.mean(hwt_data['filtered_samples']),
                np.mean(hwt_data['original_nomod']),
                np.mean(hwt_data['filtered_nomod']),
                np.mean(hwt_data['filtered_nomod']) - np.mean(hwt_data['original_nomod'])
            )
            
            # Write formatted line
            file.write(f"{hwt:3} {avg_line[1]:15.0f} {avg_line[2]:15.0f} {avg_line[3]:15.2f} {avg_line[4]:15.2f} {avg_line[5]:15.2f}\n")

# Main execution
start_time = time.time()
for threshold in thresholds:
    process_threshold(threshold)

print(f"Analysis complete. Results saved to {output_dir}/")
print(f"Total execution time: {time.time()-start_time:.2f} seconds")
