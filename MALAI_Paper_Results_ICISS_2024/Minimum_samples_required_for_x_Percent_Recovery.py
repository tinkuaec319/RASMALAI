import os

# Define the parameters
no_mod_percentages = [60, 65, 70, 75, 80]
num_samples_list = [10000, 20000, 50000, 100000, 200000, 500000, 750000, 1000000]
hamming_weights = list(range(3, 41))

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

    return success_counts

# Define the recovery rates
recovery_rates = [100, 90, 80, 70]

# Process each file and store the results
results = {no_mod: {rate: {hw: None for hw in hamming_weights} for rate in recovery_rates} for no_mod in no_mod_percentages}

for no_mod in no_mod_percentages:
    for num_samples in num_samples_list:
        filename = f"Secret_recovery_results_for_{num_samples}_samples_with_Nomod_{no_mod}%.txt"
        success_counts = process_file(filename)
        if success_counts is not None:
            for hw in hamming_weights:
                for rate in recovery_rates:
                    required_successes = rate / 10.0
                    if success_counts[hw] >= required_successes and results[no_mod][rate][hw] is None:
                        results[no_mod][rate][hw] = num_samples

# Secret_recovery_results_for_10000_samples_with_Nomod_50%
# Write the results to separate output files for each recovery rate
for rate in recovery_rates:
    output_filename = f"Minimum_samples_required_for_{rate}_Percent_Recovery.txt"
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
