import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.interpolate import make_interp_spline

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
filepath = f"NoMod_percentages_for_each_hwt_{start_hwt}__to__hwt_{end_hwt}_ternary.txt"

# ---- Read Data ---- #
hw_values, avg_percentages = extract_avg_nomod_percentages(filepath)

# ---- Filter for Multiples of 5 ---- #
marker_hw = [hw for hw in hw_values if hw % 5 == 0]
marker_avg = [avg_percentages[hw_values.index(hw)] for hw in marker_hw]

# ---- Smoothing Using Interpolation ---- #
marker_hw_np = np.array(marker_hw)
marker_avg_np = np.array(marker_avg)

# Generate a smooth curve
x_smooth = np.linspace(min(marker_hw), max(marker_hw), 300)
spline = make_interp_spline(marker_hw_np, marker_avg_np, k=3)  # Cubic spline
y_smooth = spline(x_smooth)

# ---- Plotting ---- #
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_smooth, y_smooth, linewidth=1, color='b')  # smooth line
ax.plot(marker_hw, marker_avg, marker='o', linestyle='None', markersize=5, color='b')  # markers at multiples of 5

# ---- Labels ---- #
ax.set_xlabel('Hamming weight', fontsize = 20)
ax.set_ylabel('Percentage(%) of noMod data', fontsize = 20)
# ax.set_title('Percentage of NoMod Data vs Hamming Weights', fontsize=20)

# ---- Grid & Ticks ---- #
ax.grid(which='both', linestyle='--', linewidth=0.3)
plt.xticks(np.arange(5, 41, 5))

# ---- Display Plot ---- #
plt.tight_layout()
plt.savefig(f'N_1024_Q_50_noMod_vs_hwt_ternary.png')
