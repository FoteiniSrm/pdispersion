import time
import numpy as np
import os
import csv
import argparse
from BFS_unordered_heuristic import branch_and_bound
import problems as pr

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run p-dispersion study on input files in a specified folder.")
parser.add_argument("folder_path", type=str, help="Path to the folder containing input files.")
parser.add_argument("--num_repetitions", type=int, default=1, help="Number of times to run each input (default: 1).")

args = parser.parse_args()
folder_path = args.folder_path
num_repetitions = args.num_repetitions

# Dictionaries to store results
runtime_results = {}
nodes_results = {}
best_objective_results = {}
averages = {}

# Get list of input files
input_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
input_files.sort(key=lambda x: int(x.split('.')[0]))  # Ensure numerical order

# Iterate through each file and run branch_and_bound multiple times
for file_name in input_files:
    file_path = os.path.join(folder_path, file_name)
    print(f"Processing {file_name}...")

    file_runtimes = []
    file_nodes = []
    file_best_objectives = []

    for _ in range(num_repetitions):
        # Load problem instance
        P, F, binaryConstraint, fpEucDistances, model, yi_order, max_distance, ub, lb, integer_var, num_vars, c = pr.pdispersion(file_path)

        # Initialize tracking structures
        found_pos_sol = np.zeros(num_vars, dtype=int)
        best_bound_per_depth = np.full(num_vars, -np.inf)
        nodes_per_depth = np.zeros(num_vars + 1, dtype=float)
        nodes_per_depth[0] = 1
        for i in range(1, num_vars + 1):
            nodes_per_depth[i] = nodes_per_depth[i - 1] * 2

        # Run algorithm and measure time
        start_time = time.time()
        solutions, nodes = branch_and_bound(
            yi_order, P, F, binaryConstraint, fpEucDistances, max_distance, model, ub, lb,
            integer_var, best_bound_per_depth, found_pos_sol, nodes_per_depth
        )
        end_time = time.time()
        runtime = end_time - start_time

        # Store results
        file_runtimes.append(runtime)
        file_nodes.append(nodes)
        best_objective = solutions[-1][1] if solutions else None
        file_best_objectives.append(best_objective)

    # Store results for this file
    runtime_results[file_name] = file_runtimes
    nodes_results[file_name] = file_nodes
    best_objective_results[file_name] = file_best_objectives

    # Calculate averages
    avg_runtime = np.mean(file_runtimes) if file_runtimes else None
    avg_nodes = np.mean(file_nodes) if file_nodes else None
    avg_objective = np.mean([obj for obj in file_best_objectives if obj is not None]) if file_best_objectives else None

    averages[file_name] = (avg_runtime, avg_nodes, avg_objective)

# Calculate overall averages and totals
all_runtimes = [t for runtimes in runtime_results.values() for t in runtimes if t is not None]
all_nodes = [n for nodes in nodes_results.values() for n in nodes if n is not None]
all_objectives = [obj for objectives in best_objective_results.values() for obj in objectives if obj is not None]

overall_avg_runtime = np.mean(all_runtimes) if all_runtimes else None
overall_avg_nodes = np.mean(all_nodes) if all_nodes else None
overall_avg_objective = np.mean(all_objectives) if all_objectives else None
total_runtime = np.sum(all_runtimes) if all_runtimes else None  # Summation of all runtimes
total_nodes = np.sum(all_nodes) if all_nodes else None  # Summation of all nodes

# Print computational study results
print("\n========= COMPUTATIONAL STUDY RESULTS =========")
print(f"{'File':<15}{'Runtimes (s)':<30}{'Nodes':<30}{'Best Objective':<30}{'Avg Runtime (s)':<15}{'Avg Nodes':<15}{'Avg Best Objective'}")
print("=" * 140)

for file_name in input_files:
    runtimes_str = "  ".join(f"{t:.4f}" for t in runtime_results[file_name])
    nodes_str = "  ".join(f"{n}" for n in nodes_results[file_name])
    objectives_str = "  ".join(f"{o:.4f}" if o is not None else "None" for o in best_objective_results[file_name])
    
    avg_runtime, avg_nodes, avg_objective = averages[file_name]

    print(f"{file_name:<15}{runtimes_str:<30}{nodes_str:<30}{objectives_str:<30}{avg_runtime if avg_runtime is not None else 'None':<15.4f}{avg_nodes if avg_nodes is not None else 'None':<15.4f}{avg_objective if avg_objective is not None else 'None':<15}")

# Print overall averages and total values
print("=" * 140)
print(f"{'Overall Avg':<15}{'':<30}{'':<30}{'':<30}{overall_avg_runtime if overall_avg_runtime is not None else 'None':<15.4f}{overall_avg_nodes if overall_avg_nodes is not None else 'None':<15.4f}{overall_avg_objective if overall_avg_objective is not None else 'None':<15}")
print(f"{'Total Runtime':<15}{'':<30}{'':<30}{'':<30}{total_runtime if total_runtime is not None else 'None':<15.4f}")
print(f"{'Total Nodes':<15}{'':<30}{'':<30}{'':<30}{total_nodes if total_nodes is not None else 'None':<15}")

# Export results to CSV
csv_filename = folder_path + "_bnb_study_results.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(['File', 'Runtimes (s)', 'Nodes', 'Best Objective', 'Avg Runtime (s)', 'Avg Nodes', 'Avg Best Objective'])
    
    for file_name in input_files:
        runtimes_str = "  ".join(f"{t:.4f}" for t in runtime_results[file_name])
        nodes_str = "  ".join(f"{n}" for n in nodes_results[file_name])
        objectives_str = "  ".join(f"{o:.4f}" if o is not None else "None" for o in best_objective_results[file_name])

        avg_runtime, avg_nodes, avg_objective = averages[file_name]

        writer.writerow([file_name, runtimes_str, nodes_str, objectives_str, f"{avg_runtime:.4f}", f"{avg_nodes:.4f}", f"{avg_objective if avg_objective is not None else 'None'}"])

    writer.writerow(["Overall Avg", "", "", "", f"{overall_avg_runtime:.4f}", f"{overall_avg_nodes:.4f}", f"{overall_avg_objective if overall_avg_objective is not None else 'None'}"])
    writer.writerow(["Total Runtime", "", "", "", f"{total_runtime:.4f}" if total_runtime is not None else 'None'])
    writer.writerow(["Total Nodes", "", "", "", f"{total_nodes:.4f}" if total_nodes is not None else 'None'])

print(f"\nResults saved to {csv_filename}")
