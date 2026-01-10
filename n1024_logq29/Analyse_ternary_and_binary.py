import numpy as np
import os
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit

# ================= CONFIGURATION =================
Q = 274887787
half_value = Q // 2
hamming_weight = 40
num_samples = 1000000
num_iterations = 10  # Analyze all 10 iterations

# ================= DATA LOADING =================
def load_secrets(data_type):
    base_dir = f'{data_type}_secrets_h3_40'
    return (
        np.load(os.path.join(base_dir, 'secret.npy'), allow_pickle=True),
        np.load(os.path.join(base_dir, 'train_b.npy'), allow_pickle=True),
        np.load('modified_train_A.npy', mmap_mode='r')
    )

# Load both secret types
secret_data_ternary, train_b_ternary, modified_A = load_secrets('ternary')
secret_data_binary, train_b_binary, _ = load_secrets('binary')

# ================= SAMPLE SELECTION =================
def select_samples(true_s, b_vector):
    modified_b = np.where(b_vector > half_value, b_vector - Q, b_vector)
    abs_diff = np.abs(np.dot(modified_A, true_s) - modified_b)
    mask = abs_diff < half_value
    
    no_mod_indices = np.where(mask)[0]
    mod_indices = np.where(~mask)[0]
    
    # Fixed 70% NoMod selection
    num_no_mod = int(0.7 * num_samples)
    num_mod = num_samples - num_no_mod
    
    selected = np.concatenate([
        np.random.choice(no_mod_indices, num_no_mod, False),
        np.random.choice(mod_indices, num_mod, False)
    ])
    np.random.shuffle(selected)
    return selected, modified_b[selected], mask[selected]

# ================= RECOVERY FUNCTIONS =================
def recover_ternary(model, A, b):
    model.fit(A, b)
    coef = model.coef_
    thresholds = (np.min(coef) + np.mean(coef))/2, (np.max(coef) + np.mean(coef))/2
    recovered = np.select([coef < thresholds[0], coef > thresholds[1]], [-1, 1], 0)
    return recovered, coef

def recover_binary(model, A, b):
    model.fit(A, b)
    coef = model.coef_
    threshold = (np.max(coef) + np.min(coef)) / 2
    recovered = (coef > threshold).astype(int)
    return recovered, coef

# ================= MAIN ANALYSIS =================
models = [
    ('LinearRegression', LinearRegression()),
    ('OMP', OrthogonalMatchingPursuit())
]

all_results = []

for iteration in range(num_iterations):
    # Calculate secret column for current iteration
    secret_col = (hamming_weight - 3) * 10 + iteration
    
    # Select samples using ternary parameters
    true_s_ternary = secret_data_ternary[:, secret_col]
    selected_indices, filtered_b_ternary, ternary_mask = select_samples(
        true_s_ternary, train_b_ternary[:, secret_col]
    )

    # Get binary parameters for same indices
    true_s_binary = secret_data_binary[:, secret_col]
    b_binary = train_b_binary[:, secret_col]
    modified_b_binary = np.where(b_binary > half_value, b_binary - Q, b_binary)
    filtered_b_binary = modified_b_binary[selected_indices]

    # Calculate actual NoMod% for binary
    abs_diff_binary = np.abs(np.dot(modified_A[selected_indices], true_s_binary) - filtered_b_binary)
    binary_mask = abs_diff_binary < half_value
    actual_binary_nomod = np.mean(binary_mask)

    iteration_results = []
    
    for model_name, model in models:
        # Ternary recovery
        recovered_ternary, coef_ternary = recover_ternary(
            model, modified_A[selected_indices], filtered_b_ternary
        )
        success_ternary = np.array_equal(recovered_ternary, true_s_ternary)
        
        # Binary recovery
        recovered_binary, coef_binary = recover_binary(
            model, modified_A[selected_indices], filtered_b_binary
        )
        success_binary = np.array_equal(recovered_binary, true_s_binary)
        
        iteration_results.append({
            'model': model_name,
            'ternary': {
                'success': success_ternary,
                'coef_stats': (coef_ternary.min(), coef_ternary.max(), coef_ternary.mean())
            },
            'binary': {
                'success': success_binary,
                'coef_stats': (coef_binary.min(), coef_binary.max(), coef_binary.mean()),
                'actual_nomod': actual_binary_nomod
            }
        })
    
    all_results.append({
        'iteration': iteration,
        'secret_col': secret_col,
        'results': iteration_results,
        'ternary_nomod': ternary_mask.mean(),
        'binary_nomod': actual_binary_nomod
    })

# ================= OUTPUT GENERATION =================
output = f"""Secret Recovery Analysis Report - Hamming Weight {hamming_weight}
=======================================================
Total iterations analyzed: {num_iterations}
Sample size per iteration: {num_samples}
"""

for iteration_data in all_results:
    output += f"\n\n{'='*60}"
    output += f"\nIteration {iteration_data['iteration']}"
    output += f"\nSecret column: {iteration_data['secret_col']}"
    output += f"\nTernary NoMod: {iteration_data['ternary_nomod']:.2%}"
    output += f"\nBinary NoMod: {iteration_data['binary_nomod']:.2%}"
    output += f"\n{'='*60}"
    
    for res in iteration_data['results']:
        output += f"\n\nModel: {res['model']}"
        output += f"\n---"
        output += f"\nTernary Recovery:"
        output += f"\n- Success: {'✅' if res['ternary']['success'] else '❌'}"
        output += f"\n- Coefficients (min/max/avg): {res['ternary']['coef_stats'][0]:.4f} | {res['ternary']['coef_stats'][1]:.4f} | {res['ternary']['coef_stats'][2]:.4f}"
        output += f"\n\nBinary Recovery:"
        output += f"\n- Success: {'✅' if res['binary']['success'] else '❌'}"
        output += f"\n- Coefficients (min/max/avg): {res['binary']['coef_stats'][0]:.4f} | {res['binary']['coef_stats'][1]:.4f} | {res['binary']['coef_stats'][2]:.4f}"

output += "\n\n======================================================="
output += "\nAnalysis complete"

with open(f'Analysis_of_diff_btwn_ternary_and_binary_for_HWT_{hamming_weight}_All_Iterations.txt', 'w') as f:
    f.write(output)

print(f"Analysis complete. Results saved to Recovery_Analysis_HWT{hamming_weight}_All_Iterations.txt")
