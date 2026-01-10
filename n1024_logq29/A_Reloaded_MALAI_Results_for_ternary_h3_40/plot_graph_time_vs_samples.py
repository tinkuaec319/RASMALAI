import os
import matplotlib.pyplot as plt

# Settings
no_mod_percentages = [60]
num_samples_list = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]
start_hwt = 3
end_hwt = 40
hamming_weights = range(start_hwt, end_hwt + 1)

# -------- Function to extract average time -------- #
def process_file_for_time(filename, hamming_weights):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None

    current_hamming_weight = None
    total_time = 0.0
    total_count = 0

    for line in lines:
        line = line.strip()

        # Detect Hamming weight header
        if line.startswith("****************** Hamming Weight ="):
            try:
                current_hamming_weight = int(line.split('=')[1].strip().split()[0])
            except ValueError:
                current_hamming_weight = None
                continue

        elif current_hamming_weight in hamming_weights:
            if (
                "previously recovered" in line or
                "Insufficient data" in line or
                not line or
                not line[0].isdigit()
            ):
                continue

            try:
                if "seconds" in line:
                    time_str = line.split("seconds")[0].split()[-1]  # Get the float just before "seconds"
                    time_val = float(time_str)
                    if time_val < 0:
                        continue
                    total_time += time_val
                    total_count += 1
            except (ValueError, IndexError):
                continue  # Skip malformed lines

    if total_count == 0:
        return None
    return total_time / total_count

# -------- Process all files and collect average times -------- #
average_times = []

for no_mod in no_mod_percentages:
    folder_name = f"NoMod_{no_mod}%_results_for_h{start_hwt}_{end_hwt}_ternary"

    for num_samples in num_samples_list:
        filename = os.path.join(
            folder_name,
            f"Secret_recovery_results_for_{num_samples}_samples_with_Nomod_{no_mod}%_ternary.txt"
        )
        avg_time = process_file_for_time(filename, hamming_weights)
        if avg_time is not None:
            average_times.append(avg_time)
        else:
            average_times.append(0.0)

# -------- Plotting -------- #
plt.figure(figsize=(10, 6))
x_values = [n // 10000 for n in num_samples_list]  # Convert to 10,000 scale
plt.plot(x_values, average_times, marker='o', linestyle='-', color='blue')

plt.xlabel("Number of Samples (x10,000)")
plt.ylabel("Average Recovery Time (seconds)")
plt.title("Average Secret Recovery Time vs Number of Samples (ternary, N=1024)")
plt.xticks(range(0, 101, 10))  # 0 to 100
plt.grid(True)
plt.tight_layout()
plt.savefig("Recovery_Time_vs_Samples_60.png")
plt.show()
