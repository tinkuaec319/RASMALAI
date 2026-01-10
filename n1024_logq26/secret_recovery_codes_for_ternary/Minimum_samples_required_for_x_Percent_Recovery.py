import os

# Parameters
no_mod_percentages = [60, 65, 70, 75, 80, 85, 90]
num_samples_list = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000]
start_hwt = 3
end_hwt = 40
hamming_weights = range(start_hwt, end_hwt + 1)

# Function to process files
def process_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        return None

    success_counts = {i: 0 for i in hamming_weights}
    total_counts = {i: 0 for i in hamming_weights}
    insufficient_counts = {i: 0 for i in hamming_weights}

    current_hamming_weight = None
    for line in lines:
        line = line.strip()

        if line.startswith("****************** Hamming Weight ="):
            try:
                current_hamming_weight = int(line.split('=')[1].strip().split()[0])
            except ValueError:
                current_hamming_weight = None
                continue

        elif current_hamming_weight in hamming_weights and line:
            if "Insufficient data" in line:
                insufficient_counts[current_hamming_weight] += 1
                continue
            try:
                value = int(line.split()[0])
                if value > 0:
                    success_counts[current_hamming_weight] += 1
                total_counts[current_hamming_weight] += 1
            except (IndexError, ValueError):
                continue

    return success_counts, total_counts, insufficient_counts

# Recovery rate definitions
recovery_rates = [100, 90, 80, 70, 60, 50, 10]

# Results mapping: [no_mod][rate][hamming_weight] = (num_samples, insufficient_count)
results = {
    no_mod: {
        rate: {hw: None for hw in hamming_weights}
        for rate in recovery_rates
    } for no_mod in no_mod_percentages
}

# Process all files
for no_mod in no_mod_percentages:
    folder_name = f"NoMod_{no_mod}%_results_for_h{start_hwt}_{end_hwt}_ternary"
    if not os.path.exists(folder_name):
        print(f"Folder not found: {folder_name} — skipping")
        continue

    for num_samples in num_samples_list:
        filename = os.path.join(
            folder_name,
            f"Secret_recovery_results_for_{num_samples}_samples_with_Nomod_{no_mod}%_ternary.txt"
        )
        result = process_file(filename)
        if result is None:
            continue

        success_counts, total_counts, insufficient_counts = result

        for hw in hamming_weights:
            total = total_counts[hw]
            success = success_counts[hw]
            insufficient = insufficient_counts[hw]

            for rate in recovery_rates:
                if results[no_mod][rate][hw] is not None:
                    continue  # Already satisfied with fewer samples

                if total == 0:
                    continue  # No data to evaluate recovery

                recovery_ratio = success / total
                required_ratio = rate / 100.0

                if recovery_ratio >= required_ratio:
                    # Save both sample count and insufficient count for *this* file
                    results[no_mod][rate][hw] = (num_samples, insufficient)

# Write output
for rate in recovery_rates:
    output_filename = f"Minimum_samples_required_for_{rate}_Percent_Recovery_for_h{start_hwt}_{end_hwt}_ternary.txt"
    with open(output_filename, 'w') as output_file:
        for hw in hamming_weights:
            output_file.write(f"************** Hamming weight = {hw} ****************\n")
            for no_mod in no_mod_percentages:
                entry = results[no_mod][rate][hw]
                if entry is not None:
                    num_samples, insufficient = entry
                    output_file.write(f"NoMod {no_mod}%  ==>  {num_samples} samples for {rate}% recovery")
                    if insufficient > 0:
                        output_file.write(f"  (Insufficient data = {insufficient})")
                    output_file.write("\n")
                else:
                    output_file.write(f"NoMod {no_mod}%  ==>  Not achieved for {rate}% recovery\n")
            output_file.write("\n")
    print(f"Results for {rate}% recovery have been written to {output_filename}")
