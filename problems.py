import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Create a function that reads data from a file
def read_data(filename):
    f = open(filename, "rt")
    lines = f.readlines()
    lines = [f.replace('\n', '') for f in lines if f != '\n']
    line = lines[0].split(' ')
    
    P = int(line[1])  
    F = int(line[2])
    
    line_idx = 2

    line = lines[line_idx]
    points = np.zeros(P)
    yi_order = np.zeros(P)

    for i in range(P):
        points[i] = int(lines[line_idx])
        line_idx += 1
    
    line_idx += 1
    line = lines[line_idx]
    binaryConstraint = np.zeros(shape=(F, F))
    for i in range(F - 1):
        for j in range(i + 1, F):
            line = lines[line_idx].split(' ')
            binaryConstraint[i][j] = float(line[2])
            binaryConstraint[j][i] = float(line[2])
            line_idx += 1

    line_idx += 1
    line = lines[line_idx]
    fp_euc_distances = np.zeros(shape=(P, P))
    fp_sp_distances = np.zeros(shape=(P, P))
    input_lines = []
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            
            line = lines[line_idx]
            input_lines.append(line)
            line = line.split(' ')
            fp_sp_distances[i][j] = float(line[2])
            fp_euc_distances[i][j] = float(line[3])
            line_idx += 1

    # Sort the input lines by the last value (Euclidean distance) in descending order
    sorted_lines = sorted(input_lines, key=lambda x: float(x.split()[-1]), reverse=True)

    # Extract unique first and second variables in order of appearance
    unique_vars = []
    for line in sorted_lines:
        vars_in_line = line.split()[:2]  # Get the first two variables
        for var in vars_in_line:
            if var not in unique_vars:
                unique_vars.append(var)

    # Create a mapping of points to their indices
    points_map = {str(int(val)): idx for idx, val in enumerate(points)}

    # Generate yi_order as a NumPy array
    yi_order = [points_map[var] for var in unique_vars]

    # # Generate sortedBinaryConstraints
    # sortedBinaryConstraints = []
    # for i in range(F - 1):
    #     for j in range(i + 1, F):
    #         sortedBinaryConstraints.append((i, j, binaryConstraint[i][j]))

    # # Convert to NumPy array and sort by the third column (constraint values) in descending order
    # sortedBinaryConstraints = np.array(sortedBinaryConstraints)
    # sortedBinaryConstraints = sortedBinaryConstraints[sortedBinaryConstraints[:, 2].argsort()[::-1]]

    # Max possible distance between yi and yj
    max_distance = float(sorted_lines[0].split()[3])
    

    return P, F, yi_order, binaryConstraint, max_distance, fp_sp_distances, fp_euc_distances


# Creates a model for the pdispersion problem for gurobipy
def pdispersion(filename):
    
    P, F, yi_order, binaryConstraint, max_distance, fp_sp_distances, fpEucDistances  = read_data(filename)

    # - P for yi, P*F for xij,   1 for objective 
    num_vars = P + P*F + 1

    model = gp.Model()

    # Variables xij and yi have UB=1 and LB=0
    # Objective b has UB= +inf  and LB= -inf
    ub = [1 if i < num_vars-1 else np.inf for i in range(num_vars) ]
    lb = [0 if i < num_vars-1 else -np.inf for i in range(num_vars) ]


    # Create variables
    x = model.addVars(num_vars, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="x")

    M = 1000 # A sufficiently large number to model Big M

    # Linear constraint #1 : p facilities are to be located / p sites should have a facility on them [We get p from F (Assuming F is the set of facilities to be located)]
    expr = 0
    for i in range(P):
        expr += x[i]
    model.addLConstr(expr == F)

    # Linear constraint #2 : b <= D[i,j] + M(2 - yi -yj) 
    # Open facility at site i (yi = 1) AND open facility at site j (yj = 1) implies that we have b <= D[i,j] (b less-equal to the distance between the two candidate facility locations i,j)
    expr = 0
    for i in range(P):
        for j in range(P):
            if j > i:
                expr += fpEucDistances[i][j] + M * (2 - x[i] - x[j])
                model.addLConstr(expr >= x[P + P*F])
                expr = 0

    # Linear constraint #3 : Variables xij are linked to  yi [ Facility xij ( j->facilities F   i->sites P)   is located at site  yi (i->sites P) ]
    expr = 0
    for i in range(P):
        for j in range(F):
            expr += x[P+ i*F + j]
        model.addLConstr(expr == x[i])
        expr = 0

    # Linear constraint #4 : No two variables xij and xij′ are set to 1 ( each facility site can host at most ONE facility)
    expr = 0
    for i in range(P):
        for j in range(F):
            expr += x[P + i*F + j]
        model.addLConstr(expr <= 1)
        expr = 0


    # Linear constraint #5 : No two variables xij and xi′j are set to 1 ( each facility must be hosted at exactly ONE facility site )
    expr = 0
    for i in range(F):
        for j in range(P):
            expr += x[P + j*F + i]
        model.addLConstr(expr == 1)
        expr = 0

    # linear constraint #6 : each facility should be at a safe distance from other facilities.
    expr = 0
    for i in range(P):
        for j in range(P):
            if i==j:
                continue
            for p in range(F-1):
                for l in range(p+1, F):
                    if fpEucDistances[i][j] <= binaryConstraint[p][l]:
                        model.addLConstr(x[P + i*F + p] + x[P + j*F +l] <= 1)


    # Multiply all the elements of x[i] from 0 to num_vars-1 (variables xij, yi) to to get the last variable at num_vars (objective var b)
    c = [0 if i < num_vars-1 else 1 for i in range(num_vars)]
    
    model.setObjective(gp.quicksum(c[i] * x[i] for i in range(num_vars)))
    
    model.ModelSense = GRB.MAXIMIZE # We have maximization
    model.Params.method = 1  # 1 indicates the dual Simplex algorithm in Gurobi
    model.update()

    # Save the model file
    # model.write("model.lp")
    
    # Define which variables should have integer values
    integer_var = [True if i < num_vars-1 else False for i in range(num_vars)]

    return P, F, binaryConstraint, fpEucDistances, model, yi_order, max_distance, ub, lb, integer_var, num_vars, c
    


