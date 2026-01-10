import os

# Parameters
no_mod_percentages = [60, 65, 70, 75, 80, 85, 90]
num_samples_list = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 2500000]
start_hwt = 3
end_hwt = 40
hamming_weights = range(start_hwt, end_hwt + 1)

# Function to process files
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

        # Detect block header
        if line.startswith("****************** Hamming Weight ="):
            try:
                current_hamming_weight = int(line.split('=')[1].strip().split()[0])
            except ValueError:
                current_hamming_weight = None
                continue

        elif current_hamming_weight in hamming_weights and line:
            try:
                # Extract the first token before any tabs/spaces — should be the success count
                value = int(line.split()[0])
                if value > 0:
                    success_counts[current_hamming_weight] += 1
                total_counts[current_hamming_weight] += 1
            except (IndexError, ValueError):
                continue  # Skip malformed lines

    return success_counts

# Recovery rate definitions
recovery_rates = [100, 90, 80, 70, 60, 50, 10]
results = {no_mod: {rate: {hw: None for hw in hamming_weights} for rate in recovery_rates} for no_mod in no_mod_percentages}

# Process all files
for no_mod in no_mod_percentages:
    folder_name = f"NoMod_{no_mod}%_results_for_h{start_hwt}_{end_hwt}_binary"
    if not os.path.exists(folder_name):
        print(f"Folder not found: {folder_name} — skipping")
        continue

    for num_samples in num_samples_list:
        filename = os.path.join(
            folder_name,
            f"Secret_recovery_results_for_{num_samples}_samples_with_Nomod_{no_mod}%_binary.txt"
        )
        success_counts = process_file(filename)
        if success_counts is not None:
            for hw in hamming_weights:
                for rate in recovery_rates:
                    required_successes = rate / 10.0
                    if success_counts[hw] >= required_successes and results[no_mod][rate][hw] is None:
                        results[no_mod][rate][hw] = num_samples

# Write output
for rate in recovery_rates:
    output_filename = f"Minimum_samples_required_for_{rate}_Percent_Recovery_for_h{start_hwt}_{end_hwt}_binary.txt"
    with open(output_filename, 'w') as output_file:
        for hw in hamming_weights:
            output_file.write(f"************** Hamming weight = {hw} ****************\n")
            for no_mod in no_mod_percentages:
                min_samples = results[no_mod][rate][hw]
                if min_samples is not None:
                    output_file.write(f"NoMod {no_mod}%  ==>  {min_samples} samples for {rate}% recovery\n")
                else:
                    output_file.write(f"NoMod {no_mod}%  ==>  Not achieved for {rate}% recovery\n")
            output_file.write("\n")
    print(f"Results for {rate}% recovery have been written to {output_filename}")
