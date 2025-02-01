import pyomo.environ as pyomo
import numpy as np
import time

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
    for i in range(P):
        points[i] = int(lines[line_idx])
        line_idx+=1
    
    line_idx+=1
    line = lines[line_idx]
    binaryConstraint = np.zeros(shape=(F,F))
    for i in range(F-1):
        for j in range(i+1, F):
            line = lines[line_idx].split(' ')
            binaryConstraint[i][j] = float(line[2])
            binaryConstraint[j][i] = float(line[2])
            line_idx+=1

    line_idx+=1
    line = lines[line_idx]
    fp_euc_distances = np.zeros(shape=(P,P))
    fp_sp_distances = np.zeros(shape=(P,P))
    for i in range(P):
        for j in range(P):
            if(i==j):
                continue
            line = lines[line_idx].split(' ')
            fp_sp_distances[i][j] = float(line[2])
            fp_euc_distances[i][j] = float(line[3])
            line_idx+=1


    return P,F,points,binaryConstraint,fp_sp_distances,fp_euc_distances


# Read data from a file
P,F,points,binaryConstraint,fp_sp_distances,fpEucDistances = read_data("25_3_test.txt")

# Create a concrete model
model = pyomo.ConcreteModel()


# Create range set
x_i_var_range = pyomo.RangeSet(0, P-1)
x_j_var_range = pyomo.RangeSet(0, F-1)
y_i_var_range = pyomo.RangeSet(0, P-1)

M = 5000  # A sufficiently large number to model Big M

# Create variables
model.x = pyomo.Var(x_i_var_range, x_j_var_range, domain = pyomo.Binary)
model.y = pyomo.Var(y_i_var_range, domain = pyomo.Binary)
model.b = pyomo.Var(domain = pyomo.NonNegativeReals)

# Create a constraint list
model.c = pyomo.ConstraintList()

# Linear constraint #1 : p facilities are to be located / p sites should have a facility on them [We get p from F (Assuming F is the set of facilities to be located)]
expr = 0
for i in range(P):
    expr += model.y[i]
model.c.add(expr == F)



# Linear constraint #2 : b <= D[i,j] + M(2 - yi -yj) 
# Open facility at site i (yi = 1) AND open facility at site j (yj = 1) implies that we have b <= D[i,j] (b less-equal to the distance between the two candidate facility locations i,j)
expr = 0
for i in range(P):
    for j in range(P):
        if j > i:
            expr += fpEucDistances[i][j] + M * (2 - model.y[i] - model.y[j])
            model.c.add(expr >= model.b)
            expr = 0




# Linear constraint #3 : Variables xij are linked to  yi [ Facility xij ( j->facilities F   i->sites P)   is located at site  yi (i->sites P) ]
expr = 0
for i in range(P):
    for j in range(F):
        expr += model.x[i, j]
    model.c.add(expr == model.y[i])
    expr = 0


# Linear constraint #4 : No two variables xij and xij′ are set to 1 ( each facility site can host at most ONE facility)
expr = 0
for i in range(P):
    for j in range(F):
        expr += model.x[i, j]
    model.c.add(expr <= 1)
    expr = 0


# Linear constraint #5 : No two variables xij and xi′j are set to 1 ( each facility must be hosted at exactly one facility site )
expr = 0
for i in range(F):
    for j in range(P):
        expr += model.x[j, i]
    model.c.add(expr == 1)
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
                    model.c.add(model.x[i, p] + model.x[j, l] <= 1)


# Create objective 
model.objective = pyomo.Objective(sense = pyomo.maximize, expr = model.b)

# Print the model
# model.pprint()
# model.write("model1.lp") 

# Create solver object and choose one (e.g. Gurobi)
solver = pyomo.SolverFactory('gurobi')

# Solve/Model the model
start = time.time()
result = solver.solve(model)
end = time.time()
# Print results 
print(result)
print("Print values for all variables")

# Access all variables of our model
for v in model.component_data_objects(pyomo.Var):
  print(str(v), v.value)
print(f"Objective value = {pyomo.value(model.objective)}")
print(f"Time Elapsed: {end - start}")

    