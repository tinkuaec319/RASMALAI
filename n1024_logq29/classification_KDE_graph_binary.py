import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Set font size for all elements to 16
plt.rcParams.update({'font.size': 16})
sns.set_context("notebook", font_scale=1.5)

# ================= CONFIGURATION =================
start_hwt = 3
end_hwt = 40
Q = 274887787
half_value = Q // 2
hamming_weights = [40]

# ================= DATA LOADING =================
secret_data = np.load(f'binary_secrets_h{start_hwt}_{end_hwt}/secret.npy', allow_pickle=True)
train_b = np.load(f'binary_secrets_h{start_hwt}_{end_hwt}/train_b.npy', allow_pickle=True)
modified_A = np.load('modified_train_A.npy', mmap_mode='r')

# Format x-axis ticks as multiples of Q
def x_axis_formatter(x, pos):
    q_ratio = x / Q
    if np.isclose(q_ratio, 0):
        return '0'
    elif q_ratio > 0:
        return f'{int(q_ratio)}Q'
    else:
        return f'{int(q_ratio)}Q'

# ================= PLOTTING =================
for hamming_weight in hamming_weights:
    initial_column = (hamming_weight - 3) * 10
    true_s_column = initial_column
    b_column = true_s_column
    true_s = secret_data[:, true_s_column]
    b = train_b[:, b_column]
    modified_b = np.where(b > half_value, b - Q, b)

    # Shift A by subtracting modified_b/hamming_weight
    A_shifted = modified_A - (modified_b / hamming_weight)[:, np.newaxis]

    # Calculate modified_A * true_s
    result = np.dot(modified_A, true_s)
    abs_diff = np.abs(result - modified_b)
    no_mod_indices = np.where(abs_diff < half_value)[0]
    mod_indices = np.where(abs_diff >= half_value)[0]

    # Convert A_shifted matrix to a vector by summing each row
    vector_A_shifted = np.sum(A_shifted, axis=1)

    # Plot the histogram with KDE for both NoMod and Mod data
    plt.figure(figsize=(14, 8))

    # KDE plot for NoMod and Mod
    sns.kdeplot(vector_A_shifted[no_mod_indices], label='NoMod', color='royalblue', fill=True)
    sns.kdeplot(vector_A_shifted[mod_indices], label='Mod', color='darkorange', fill=True)

    # Add labels and title
    plt.xlabel('Summation of All Columns of A')
    plt.ylabel('Kernel Density')
    plt.title(f'Distribution of Summation of Columns of A for HWt = {hamming_weight}_binary')
    plt.legend()

    # Add vertical lines at tick positions
    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    
    # Set major ticks at every 5Q multiple
    tick_interval = 10 * Q
    start_tick = int(np.floor(x_min / tick_interval)) * tick_interval
    end_tick = int(np.ceil(x_max / tick_interval)) * tick_interval
    ticks = np.arange(start_tick, end_tick + tick_interval, tick_interval)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))

    for pos in ticks:
        plt.axvline(x=pos, color='black', linestyle='--', linewidth=0.8)

    # Save the plot
    output_image_file = f'Kernel_Density_Estimation_KDE_hwt_{hamming_weight}_binary.png'
    plt.savefig(output_image_file)
    plt.close()

print("KDE plot generated and saved.")
