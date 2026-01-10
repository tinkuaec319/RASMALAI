import matplotlib.pyplot as plt
import numpy as np
import os
import re

# ================= CONFIGURATION =================
start_hwt = 3
end_hwt = 40
no_mod_percentages = [60, 65, 70, 75, 80]
num_samples_list = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]
hamming_weights = list(range(20, 41))
base_dir = "."

# Visualization parameters
font_size = 16          # General font size
label_font_size = 18    # Axis labels
legend_font_size = 18   # Legend text
figure_size = (12, 8)   # Width, height in inches
colors = ['b', 'g', 'r', 'c', 'm']

def process_result_file(file_path, debug=False):
    """Process files with robust parsing and detailed statistics"""
    if not os.path.exists(file_path):
        if debug: print(f"File not found: {file_path}")
        return None, None

    hw_results = {}
    current_hwt = None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                
                # Detect Hamming weight headers
                if re.search(r'hamming\s*weight\s*=\s*(\d+)', line, re.IGNORECASE):
                    if hw_match := re.search(r'(\d+)', line):
                        current_hwt = int(hw_match.group(1))
                        if current_hwt in hamming_weights:
                            hw_results[current_hwt] = {'successes': 0, 'total': 0}
                
                # Process result lines
                elif current_hwt and current_hwt in hamming_weights:
                    if 'insufficient data' in line.lower(): continue
                    if parts := [p.strip() for p in re.split(r'\t+', line) if p.strip()]:
                        if parts[0].isdigit():
                            result = int(parts[0])
                            hw_results[current_hwt]['total'] += 1
                            if result > 0:
                                hw_results[current_hwt]['successes'] += 1

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

    # Calculate success rates
    valid_rates = []
    for hw in hamming_weights:
        if hw in hw_results and hw_results[hw]['total'] > 0:
            rate = (hw_results[hw]['successes'] / hw_results[hw]['total']) * 100
            valid_rates.append(rate)
    
    overall_rate = np.mean(valid_rates) if valid_rates else 0.0
    return overall_rate, hw_results

def analyze_and_plot():
    """Main analysis and visualization function"""
    results = {nomod: [] for nomod in no_mod_percentages}
    
    for nomod in no_mod_percentages:
        folder_name = f"NoMod_{nomod}%_results_for_h{start_hwt}_{end_hwt}_binary"
        folder_path = os.path.join(base_dir, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue
            
        for samples in num_samples_list:
            file_name = f"Secret_recovery_results_for_{samples}_samples_with_Nomod_{nomod}%_binary.txt"
            file_path = os.path.join(folder_path, file_name)
            
            success_rate, _ = process_result_file(file_path)
            if success_rate is not None:
                results[nomod].append((samples // 10000, success_rate))

    # Create visualization with specified formatting
    plt.figure(figsize=figure_size)
    
    for idx, nomod in enumerate(no_mod_percentages):
        data_points = results[nomod]
        if data_points:
            samples_x, rates_y = zip(*sorted(data_points))
            plt.plot(samples_x, rates_y,
                     label=f'NoMod {nomod}%',
                     color=colors[idx],
                     linewidth=2.5)  # Removed markers

    # Formatting with user-specified parameters
    plt.xlabel('Number of Samples (x 10,000)', fontsize=label_font_size)
    plt.ylabel('Percentage(%) of Secret Recovery', fontsize=label_font_size)
    
    plt.legend(fontsize=legend_font_size, loc='upper left', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.xticks(np.arange(0, 101, 10), fontsize=font_size)
    plt.yticks(np.arange(0, 101, 10), fontsize=font_size)
    plt.xlim(0.5, max(num_samples_list)//10000 + 5)
    plt.ylim(-5, 105)

    plt.tight_layout()
    plt.savefig("Binary_Final_till_1M_NoMod_80.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_and_plot()
