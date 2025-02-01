import numpy as np
import random

# Definition of random heuristic for p-dispersion
def random_heuristic(max_iterations, P,F, fp_euc_distances, binaryConstraint):

    tempsol = np.zeros(F, dtype=int)
    best_cost = np.inf
    val_selected = -1
    infeasible = True
    iterations = 0

    # For every variable/facility try to assign the best value/facilitypoint having only a local view of the problem
    while(infeasible and iterations < max_iterations):
        used_values = []
        val_selected = -1
        tempsol = np.zeros(F, dtype=int)
        for i in range(F):
            val_selected = random.randint(0,P-1)
            while(val_selected in used_values):
                val_selected = random.randint(0,P-1)
            
            tempsol[i] = val_selected
            used_values.append(val_selected)    
        status = check_feasibility(tempsol, F, binaryConstraint, fp_euc_distances)
        if status:
            infeasible = False
        
        iterations += 1

    if infeasible == True:
        return False, [], 0
    best_cost = getAssignmentCost(tempsol, F, fp_euc_distances)

    return True, tempsol, best_cost


# Definition of Multiple Random Runs heuristic for p-center
def random_multiple_runs_heuristic(runs, max_iterations, P, F, fp_euc_distances, binaryConstraint):
    
    best_cost = -np.inf
    best_solution = None
    best_status = False
    
    for _ in range(runs):
        status, solution, cost = random_heuristic(max_iterations,P,F, fp_euc_distances, binaryConstraint)
        if status and cost > best_cost:
            best_cost = cost
            best_solution = solution
            best_status = status

    return best_status, best_solution, best_cost


# Function that calculates the cost of a partial/complete assignment (implementation for p-dispersion)
def getAssignmentCost(tempsol, F, fp_euc_distances):
    min_distance = np.inf
    
    # Calculate the minimum distance between any two selected facilities
    for i in range(F):
        for j in range(i+1, F):
            cost = fp_euc_distances[tempsol[i], tempsol[j]]
            if min_distance > cost:
                min_distance = cost
    
    return min_distance

# Function that checks if the given partial/complete assignment is feasible
def check_feasibility(solution, F, binaryCstrs, fp_euc_distances):

    # Check distance constraints between facilities
    for i in range(F-1):
        for j in range(i+1, F):
            # Check if the distance between two selected facilities violates any binary constraints
            if fp_euc_distances[solution[i], solution[j]] <= binaryCstrs[i][j]:
                return False
    
    return True

