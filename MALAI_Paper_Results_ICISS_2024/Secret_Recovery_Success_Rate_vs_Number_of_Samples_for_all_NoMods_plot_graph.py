import matplotlib.pyplot as plt
import numpy as np

# Define the parameters
no_mod_percentages = [60, 65, 70, 75, 80]
num_samples_list = [10000, 20000, 50000, 100000, 200000, 500000, 750000, 1000000]
hamming_weights = list(range(15, 41))

# Function to read and process a file
def process_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None
    
    success_counts = {i: 0 for i in hamming_weights}
    total_counts = {i: 0 for i in hamming_weights}

    current_hamming_weight = None
    for line in lines:
        line = line.strip()
        if line.startswith("******************hamming weight ="):
            current_hamming_weight = int(line.split('=')[1].strip().split()[0])
        elif line.isdigit():
            value = int(line)
            if current_hamming_weight in hamming_weights:
                if value > 0:
                    success_counts[current_hamming_weight] += 1
                total_counts[current_hamming_weight] += 1

    success_rates = {hw: (success_counts[hw] / total_counts[hw]) if total_counts[hw] > 0 else 0 for hw in hamming_weights}
    overall_success_rate = sum(success_counts.values()) / sum(total_counts.values()) if sum(total_counts.values()) > 0 else 0
    return overall_success_rate

# Read and process each file
results = {no_mod: [] for no_mod in no_mod_percentages}
for no_mod in no_mod_percentages:
    for num_samples in num_samples_list:
        filename = f"Secret_recovery_results_for_{num_samples}_samples_with_Nomod_{no_mod}%.txt"
        success_rate = process_file(filename)
        if success_rate is not None:
            results[no_mod].append((num_samples // 10000, success_rate * 100))  # Convert samples to 10k units and success rate to percentage

# Font size settings
font_size = 16  # Increase the font size
label_font_size = 18  # Font size for axes labels
legend_font_size = 18  # Font size for legend

# Plot the results
plt.figure(figsize=(12, 8))  # Increase figure size
colors = ['b', 'g', 'r', 'c', 'm']
for i, no_mod in enumerate(no_mod_percentages):
    data = results[no_mod]
    if data:
        num_samples, success_rates = zip(*data)
        plt.plot(num_samples, success_rates, label=f'noMod {no_mod}%', color=colors[i])

plt.xlabel('Number of Samples (in 10,000s)', fontsize=label_font_size)
plt.ylabel('Success Rate of Secret Recovery (%)', fontsize=label_font_size)
plt.legend(fontsize=legend_font_size, loc='lower right')  # Adjust legend size and location
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Customize the ticks for a cleaner look
plt.xticks(np.arange(1, 101, 10), fontsize=font_size)  # Larger X-ticks
plt.yticks(np.arange(0, 101, 10), fontsize=font_size)  # Larger Y-ticks
plt.minorticks_off()  # Turn off minor ticks to match the second image

# Save the updated plot
plt.savefig("Updated_Secret_Recovery_Success_Rate_vs_Number_of_Samples.png")
plt.show()
