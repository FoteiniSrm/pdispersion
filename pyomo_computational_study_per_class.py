import time
import numpy as np
import os
import csv
import argparse
from pyomo_pdispersion import solve_pdispersion_pyomo

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run p-dispersion study on input files in a specified folder.")
parser.add_argument("folder_path", type=str, help="Path to the folder containing input files.")
parser.add_argument("--num_repetitions", type=int, default=1, help="Number of times to run each input (default: 1).")

args = parser.parse_args()
folder_path = args.folder_path
num_repetitions = args.num_repetitions

# Dictionaries to store results
runtime_results = {}
best_objective_results = {}
averages = {}

# Get list of input files
input_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
input_files.sort(key=lambda x: int(x.split('.')[0]))  # Ensure numerical order

# Iterate through each file and run multiple times
for file_name in input_files:
    file_path = os.path.join(folder_path, file_name)
    print(f"Processing {file_name}...")

    file_runtimes = []
    file_best_objectives = []

    for _ in range(num_repetitions):
        start_time = time.time()
        
        try:
            st, tc, obj, solving_time = solve_pdispersion_pyomo(file_path)
            file_runtimes.append(solving_time)
            file_best_objectives.append(obj)
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    # Store results
    runtime_results[file_name] = file_runtimes
    best_objective_results[file_name] = file_best_objectives

    # Calculate averages
    avg_runtime = np.mean(file_runtimes) if file_runtimes else None
    avg_objective = np.mean([obj for obj in file_best_objectives if obj is not None]) if file_best_objectives else None
    averages[file_name] = (avg_runtime, avg_objective)

# Calculate overall averages and total runtime
all_runtimes = [t for runtimes in runtime_results.values() for t in runtimes if t is not None]  # Flatten runtime lists
all_objectives = [obj for objectives in best_objective_results.values() for obj in objectives if obj is not None]

overall_avg_runtime = np.mean(all_runtimes) if all_runtimes else None
overall_avg_objective = np.mean(all_objectives) if all_objectives else None
total_runtime = np.sum(all_runtimes) if all_runtimes else None  # Summation of all runtimes

# Print computational study results
print("\n========= COMPUTATIONAL STUDY RESULTS =========")
print(f"{'File':<15}{'Runtimes (s)':<30}{'Best Objective':<30}{'Avg Runtime (s)':<15}{'Avg Best Objective'}")
print("=" * 120)

for file_name in input_files:
    runtimes_str = "  ".join(f"{t:.4f}" for t in runtime_results[file_name])
    objectives_str = "  ".join(f"{o:.4f}" if o is not None else "None" for o in best_objective_results[file_name])
    
    avg_runtime, avg_objective = averages[file_name]

    print(f"{file_name:<15}{runtimes_str:<30}{objectives_str:<30}{avg_runtime if avg_runtime is not None else 'None':<15.4f}{avg_objective if avg_objective is not None else 'None':<15}")

# Print overall averages and total runtime
print("=" * 120)
print(f"{'Overall Avg':<15}{'':<30}{'':<30}{'':<30}{overall_avg_runtime if overall_avg_runtime is not None else 'None':<15.4f}{overall_avg_objective if overall_avg_objective is not None else 'None':<15}")
print(f"{'Total Runtime':<15}{'':<30}{'':<30}{'':<30}{total_runtime if total_runtime is not None else 'None':<15.4f}")

# Export results to CSV
csv_filename = folder_path + "_pyomo_study_results.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(['File', 'Runtimes (s)', 'Best Objective', 'Avg Runtime (s)', 'Avg Best Objective'])
    
    for file_name in input_files:
        runtimes_str = "  ".join(f"{t:.4f}" for t in runtime_results[file_name])
        objectives_str = "  ".join(f"{o:.4f}" if o is not None else "None" for o in best_objective_results[file_name])

        avg_runtime, avg_objective = averages[file_name]
        writer.writerow([file_name, runtimes_str, objectives_str, f"{avg_runtime:.4f}", f"{avg_objective if avg_objective is not None else 'None'}"])

    writer.writerow(["Overall Avg", "", "", "", f"{overall_avg_runtime:.4f}", f"{overall_avg_objective if overall_avg_objective is not None else 'None'}"])
    writer.writerow(["Total Runtime", "", "", "", f"{total_runtime:.4f}" if total_runtime is not None else 'None'])

print(f"\nResults saved to {csv_filename}")
