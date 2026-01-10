import numpy as np
from numpy.linalg import matrix_rank
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit
import os

# ================= CONFIGURATION =================
start_hwt = 3
end_hwt = 40
nomod_percentage = 0.60  # Binary recovery
Q = 274887787
half_value = Q // 2

models = [LinearRegression(), OrthogonalMatchingPursuit()]

output_dir = f"Analysis_for_difference_between_10_and_100_percent_recovery_binary"
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(
    output_dir, 
    f"Analysis_for_difference_between_10_and_100_percent_recovery_for_Nomod_{nomod_percentage*100:.0f}%_binary.txt"
)

# ================= DATA LOADING =================
secret_data = np.load(f'binary_secrets_h{start_hwt}_{end_hwt}/secret.npy', allow_pickle=True)
train_b = np.load(f'binary_secrets_h{start_hwt}_{end_hwt}/train_b.npy', allow_pickle=True)
modified_A = np.load('modified_train_A.npy', mmap_mode='r')

# ================= CORE RECOVERY LOGIC =================
def recover_secret(model, filtered_A, filtered_b):
    """Your original working recovery function with thresholding"""
    model.fit(filtered_A, filtered_b)
    recovered_s = model.coef_ if hasattr(model, 'coef_') else model.feature_importances_
    threshold = (np.max(recovered_s) + np.min(recovered_s)) / 2
    return (recovered_s > threshold).astype(int)

def track_recovery(true_s, modified_A, modified_b, selected_indices):
    """Track successful recoveries using your working method"""
    iter_success = 0
    for model in models:
        recovered = recover_secret(model, 
                                 modified_A[selected_indices], 
                                 modified_b[selected_indices])
        if np.array_equal(recovered, true_s):
            iter_success += 1
    return iter_success

# ================= ANALYSIS FUNCTIONS =================
def analyze_samples(A, b, true_s):
    """Compute comprehensive sample analysis metrics"""
    error = b - np.dot(A, true_s)
    return {
        'matrix_rank': matrix_rank(A),
        'dependent_samples': A.shape[0] - matrix_rank(A),
        'avg_error': np.mean(np.abs(error)),
        'error_variance': np.var(error),
        'max_error': np.max(np.abs(error)),
        'sparsity_ratio': np.mean(true_s != 0)
    }

# ================= MAIN COMPARISON LOGIC =================
def process_comparisons(easy_cases, difficult_cases):
    """Main analysis engine with file output"""
    with open(output_filename, 'w') as f:
        f.write("LWE Secret Recovery Analysis Report\n")
        f.write("===================================\n\n")
        
        for idx, (easy, diff) in enumerate(zip(easy_cases, difficult_cases)):
            easy_hwt, easy_iter, easy_samples = easy
            diff_hwt, diff_iter, diff_samples = diff

            # Calculate secret columns
            s_col_easy = (easy_hwt - start_hwt) * 10 + easy_iter
            s_col_diff = (diff_hwt - start_hwt) * 10 + diff_iter

            # Load secrets and b vectors
            true_s_easy = secret_data[:, s_col_easy]
            true_s_diff = secret_data[:, s_col_diff]
            
            # Preprocess b values
            modified_b_easy = np.where(train_b[:, s_col_easy] > half_value, 
                                      train_b[:, s_col_easy] - Q, 
                                      train_b[:, s_col_easy])
            modified_b_diff = np.where(train_b[:, s_col_diff] > half_value, 
                                      train_b[:, s_col_diff] - Q, 
                                      train_b[:, s_col_diff])

            # Generate sample indices
            def get_indices(true_s, modified_b, samples):
                abs_diff = np.abs(np.dot(modified_A, true_s) - modified_b)
                mask = abs_diff < half_value
                no_mod = np.where(mask)[0]
                mod = np.where(~mask)[0]
                num_no_mod = int(nomod_percentage * samples)
                return np.random.permutation(np.concatenate([
                    np.random.choice(no_mod, num_no_mod, False),
                    np.random.choice(mod, samples - num_no_mod, False)
                ]))

            easy_indices = get_indices(true_s_easy, modified_b_easy, easy_samples)
            diff_indices = get_indices(true_s_diff, modified_b_diff, diff_samples)

            # Track recoveries
            easy_success = track_recovery(true_s_easy, modified_A, modified_b_easy, easy_indices)
            diff_success = track_recovery(true_s_diff, modified_A, modified_b_diff, diff_indices)

            # Write comparison header
            f.write(f"\nComparison {idx+1}:\n")
            f.write(f"  Easy (hwt={easy_hwt}, iter={easy_iter}, samples={easy_samples}): {easy_success}/{len(models)} models\n")
            f.write(f"  Difficult (hwt={diff_hwt}, iter={diff_iter}, samples={diff_samples}): {diff_success}/{len(models)} models\n")

            # Perform analysis only when conditions met
            if easy_success > 0 and diff_success == 0:
                easy_metrics = analyze_samples(modified_A[easy_indices], 
                                             modified_b_easy[easy_indices], 
                                             true_s_easy)
                diff_metrics = analyze_samples(modified_A[diff_indices],
                                             modified_b_diff[diff_indices],
                                             true_s_diff)

                f.write("\n  Comparative Analysis:\n")
                f.write("  ----------------------\n")
                f.write("  Metric                | Easy Case       | Difficult Case\n")
                f.write("  ---------------------------------------------------------\n")
                for metric in easy_metrics:
                    easy_val = f"{easy_metrics[metric]:.4f}"
                    diff_val = f"{diff_metrics[metric]:.4f}"
                    f.write(f"  {metric:20} | {easy_val:14} | {diff_val:14}\n")
            else:
                f.write("\n  Analysis skipped - conditions not met\n")
            
            f.write("\n" + "="*60 + "\n\n\n\n\n")

# ================= EXAMPLE USAGE =================
if __name__ == "__main__":
    # Format: (hwt, iteration, samples)
    easy_to_recover = [
        (7,  1, 200000),
        (4,  1, 200000),
        (7,  3, 200000),
        (22, 5, 1000000),
        (24, 7, 1000000),
        (26, 2, 2000000),
        (40, 9, 1000000)
    ]
    
    difficult_to_recover = [
        (7,  8, 200000),
        (4,  5, 200000),
        (7,  5, 200000),
        (22, 1, 1000000),
        (24, 4, 1000000),
        (26, 8, 2000000),
        (40, 2, 1000000)
    ]
    
    process_comparisons(easy_to_recover, difficult_to_recover)
    print(f"Analysis complete. Results saved to {output_filename}")
