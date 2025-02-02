import pyomo.environ as pyomo
import numpy as np

TIME_LIMIT = 3600

def read_data(filename):
    """Reads input data from a given filename and returns relevant matrices and parameters."""
    with open(filename, "rt") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    P, F = map(int, lines[0].split()[1:3])  # Number of points (P) and facilities (F)

    line_idx = 2
    points = np.array([int(lines[i]) for i in range(line_idx, line_idx + P)])
    
    line_idx += P + 1
    binaryConstraint = np.zeros((F, F))
    
    for i in range(F-1):
        for j in range(i+1, F):
            binaryConstraint[i][j] = float(lines[line_idx].split()[2])
            binaryConstraint[j][i] = binaryConstraint[i][j]  # Symmetric matrix
            line_idx += 1

    line_idx += 1
    fp_euc_distances = np.zeros((P, P))
    fp_sp_distances = np.zeros((P, P))
    
    for i in range(P):
        for j in range(P):
            if i != j:
                _, _, sp_dist, euc_dist = map(float, lines[line_idx].split())
                fp_sp_distances[i][j] = sp_dist
                fp_euc_distances[i][j] = euc_dist
                line_idx += 1

    return P, F, points, binaryConstraint, fp_sp_distances, fp_euc_distances


def solve_pdispersion_pyomo(filename):
    """Reads data, builds the optimization model, solves it, and returns results."""
    
    # Read data from file
    P, F, points, binaryConstraint, fp_sp_distances, fpEucDistances = read_data(filename)

    # Create a concrete model
    model = pyomo.ConcreteModel()

    # Define range sets
    x_i_var_range = pyomo.RangeSet(0, P-1)
    x_j_var_range = pyomo.RangeSet(0, F-1)
    y_i_var_range = pyomo.RangeSet(0, P-1)

    M = 5000  # Large constant for Big M formulation

    # Define variables
    model.x = pyomo.Var(x_i_var_range, x_j_var_range, domain=pyomo.Binary)
    model.y = pyomo.Var(y_i_var_range, domain=pyomo.Binary)
    model.b = pyomo.Var(domain=pyomo.NonNegativeReals)

    # Define constraints
    model.c = pyomo.ConstraintList()

    # Constraint #1: Exactly F facilities should be located
    model.c.add(sum(model.y[i] for i in range(P)) == F)

    # Constraint #2: Facilities should maintain safe distance
    for i in range(P):
        for j in range(P):
            if j > i:
                model.c.add(model.b <= fpEucDistances[i][j] + M * (2 - model.y[i] - model.y[j]))

    # Constraint #3: Facilities are assigned to open sites
    for i in range(P):
        model.c.add(sum(model.x[i, j] for j in range(F)) == model.y[i])

    # Constraint #4: Each facility is located at exactly one site
    for i in range(F):
        model.c.add(sum(model.x[j, i] for j in range(P)) == 1)

    # Constraint #5: No facility site hosts more than one facility
    for i in range(P):
        model.c.add(sum(model.x[i, j] for j in range(F)) <= 1)

    # Constraint #6: Facility placement respects binary constraints
    for i in range(P):
        for j in range(P):
            if i != j:
                for p in range(F-1):
                    for l in range(p+1, F):
                        if fpEucDistances[i][j] <= binaryConstraint[p][l]:
                            model.c.add(model.x[i, p] + model.x[j, l] <= 1)

    # Define objective function (maximize minimum facility distance)
    model.objective = pyomo.Objective(sense=pyomo.maximize, expr=model.b)

    # Create solver object and choose one (e.g. Gurobi).
    solver = pyomo.SolverFactory('gurobi_persistent')
    solver.set_instance(model)

    # Set timeout of 1 hour for Gurobi
    solver.options['TimeLimit'] = TIME_LIMIT

    
    # Solve/Model the model
    results = solver.solve(tee=True)

    # Check if the model is infeasible
    if results.solver.status == pyomo.SolverStatus.ok:
        if results.solver.termination_condition == pyomo.TerminationCondition.optimal:
            print("Model solved to optimality")
            print(f"Objective value = {pyomo.value(model.objective)}\n\n")
            return results.solver.status, results.solver.termination_condition, pyomo.value(model.objective), results.solver.wallclock_time
        else:
            print("Solver terminated with condition:", results.solver.termination_condition)
            print(f"Objective value = {pyomo.value(model.objective)}\n\n")
            return results.solver.status, results.solver.termination_condition, pyomo.value(model.objective), results.solver.wallclock_time
    else:
        if results.solver.termination_condition == pyomo.TerminationCondition.infeasible:
            print("Model is infeasible\n\n")
            return results.solver.status, results.solver.termination_condition, -1, results.solver.wallclock_time
        else:
            print("Solver status:\n\n", results.solver.status)
            return results.solver.status, results.solver.termination_condition, -1, results.solver.wallclock_time
    
