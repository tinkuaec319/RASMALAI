import matplotlib.pyplot as plt
import numpy as np
import re

start_hwt = 3
end_hwt = 40

def extract_avg_nomod_percentages(filepath):
    """
    Extracts average NoMOD % for each Hamming weight from the given file.
    """
    hamming_weights = []
    averages = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            match = re.match(r"Average NoMOD % for HW=(\d+): ([\d.]+)%", line)
            if match:
                hw = int(match.group(1))
                avg_percent = float(match.group(2))
                hamming_weights.append(hw)
                averages.append(avg_percent)

    return hamming_weights, averages

# ---- Configuration ---- #
filepath = f"NoMod_percentages_for_each_hwt_{start_hwt}__to__hwt_{end_hwt}_binary.txt"

# ---- Read Data ---- #
hw_values, avg_percentages = extract_avg_nomod_percentages(filepath)

# ---- Plotting ---- #
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(hw_values, avg_percentages, marker='o', markersize=5, linestyle='-', linewidth=1, color='b')

# ---- Labels ---- #
ax.set_xlabel('Hamming Weight', fontsize=20)
ax.set_ylabel('Percentage (%) of NoMod data', fontsize=20)
# ax.set_title('Percentage of NoMod Data vs Hamming Weights', fontsize=20)

# ---- Grid & Ticks ---- #
ax.grid(which='both', linestyle='--', linewidth=0.3)
plt.xticks(np.arange(min(hw_values), max(hw_values) + 1, 1))

# ---- Display Plot ---- #
plt.tight_layout()
plt.savefig(f'NoMod_Percentage_vs_Hamming_Weight_h{start_hwt}_{end_hwt}_binary.png')

