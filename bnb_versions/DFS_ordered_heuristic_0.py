from gurobipy import GRB
import numpy as np
import time
import heuristics as heur
from collections import deque
import problems as pr

# define global variables
DEBUG_MODE = False # debug enabled / disabled
lower_bound = -np.inf # lower bound of the problem
upper_bound = np.inf # upper bound of the problems

# calculate if a value is very close to an integer value
def is_nearly_integer(value, tolerance=1e-6):
    return abs(value - round(value)) <= tolerance

# a class 'Node' that holds information of a node
class Node:
    def __init__(self, ub, lb, depth, vbasis, cbasis, branching_var, label=""):
        self.ub = ub
        self.lb = lb
        self.depth = depth
        self.vbasis = vbasis
        self.cbasis = cbasis
        self.branching_var = branching_var
        self.label = label

# print debugging info
def debug_print(node: Node = None, x_obj=None, sol_status=None):
    print("\n\n-----------------  DEBUG OUTPUT  -----------------\n\n")
    print(f"UB:{upper_bound}")
    print(f"LB:{lower_bound}")
    if node is not None:
        print(f"Branching Var: {node.branching_var}")
    if node is not None:
        print(f"Child: {node.label}")
    if node is not None:
        print(f"Depth: {node.depth}")
    if x_obj is not None:
        print(f"Simplex Objective: {x_obj}")
    if sol_status is not None:
        print(f"Solution status: {sol_status}")

    # print("\n\n--------------------------------------------------\n\n")


# branch & bound algorithm
def branch_and_bound(yi_order, P, F, binaryConstraint, fpEucDistances, max_distance, model, ub, lb, integer_var, best_bound_per_depth, found_pos_sol, nodes_per_depth, vbasis=[], cbasis=[], depth=0):
    global nodes, lower_bound, upper_bound

    # create stack using deque() structure
    stack = deque()

    # initialize solution list
    solutions = list()
   
    # create root node
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")
    nodes_per_depth[0] -= 1

    # Call the heuristic function
    status, solution, cost = heur.random_heuristic(5, P, F, fpEucDistances, binaryConstraint)

    # ===============  Root node  ==========================
    if DEBUG_MODE:
        debug_print()

    # solve relaxed problem
    model.setParam('OutputFlag', 0)
    model.optimize()

    nodes = 1
    
    # check if the model was solved to optimality. If not then return (infeasible).
    if model.status != GRB.OPTIMAL:
        if DEBUG_MODE:
            debug_print(node=root_node, sol_status="Infeasible")
        return [], nodes
    

    # get the solution (variable assignments)
    x_candidate = model.getAttr('X', model.getVars())

    # get the objective value
    x_obj = model.ObjVal
    best_bound_per_depth[0] = x_obj

    # check if all variables have integer values (from the ones that are supposed to be integers).
    # If not, then select the first variable with a fractional value to be the one fixed
    vars_have_integer_vals = True
    for idx, is_int_var in enumerate(integer_var):
        if is_int_var and idx < len(yi_order) and not is_nearly_integer(x_candidate[yi_order[idx]]):
            vars_have_integer_vals = False
            selected_var_idx = yi_order[idx]
            break
        elif is_int_var and idx > len(yi_order) and not is_nearly_integer(x_candidate[idx]):                   
            vars_have_integer_vals = False
            selected_var_idx = idx
            break

    # found feasible solution
    if vars_have_integer_vals:
        # if we have a feasible solution in root, then terminate
        solutions.append([x_candidate, x_obj, depth])

        if DEBUG_MODE:
            debug_print(node=root_node, x_obj=x_obj, sol_status="Integer")
        return solutions, nodes

    # otherwise update lower/upper bound for min/max respectively
    else:    
        upper_bound = x_obj
        

    if DEBUG_MODE:
        debug_print(node=root_node, x_obj=x_obj, sol_status="Fractional")

    # Retrieve vbasis and cbasis
    vbasis = model.getAttr("VBasis", model.getVars())
    cbasis = model.getAttr("CBasis", model.getConstrs())

    # create lower bounds and upper bounds for the variables of the child nodes
    left_lb = np.copy(lb)
    left_ub = np.copy(ub)
    right_lb = np.copy(lb)
    right_ub = np.copy(ub)

    # create left and right branches (e.g. set left: x = 0, right: x = 1 in a binary problem)
    left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
    right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

    # create child nodes
    left_child = Node(left_ub, left_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
    right_child = Node(right_ub, right_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

    # add child nodes in stack
    stack.append(right_child)
    stack.append(left_child)

    

    # solving sub problems
    # While the stack has nodes, continue solving
    while (len(stack) != 0):
        
        if DEBUG_MODE:
            print("\n********************************  NEW NODE BEING EXPLORED  ******************************** ")

        # increment total nodes by 1
        nodes += 1

        # get the child node on top of stack
        current_node = stack[-1]

        # remove this node from stack
        stack.pop()

        # increase the nodes visited for current depth
        nodes_per_depth[current_node.depth] -= 1

        # warm start solver. Use the vbasis and cbasis that parent node passed to the current one.
        if (len(current_node.vbasis) != 0) and (len(current_node.cbasis) != 0):
            model.setAttr("VBasis", model.getVars(), current_node.vbasis)
            model.setAttr("CBasis", model.getConstrs(), current_node.cbasis)

        # update the state of the model, passing the new lower bounds/upper bounds for the vars.
        # Basically, we only change the ub/lb for the branching variable. Another way is to introduce a new constraint (e.g. x_i <= ub).
        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)
        model.update()

        if DEBUG_MODE:
            debug_print()

        # optimize the model
        model.optimize()

        # Check if the model was solved to optimality. If not then do not create child nodes.
        infeasible = False
        if model.status != GRB.OPTIMAL:        
            infeasible = True
            x_obj = -np.inf
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)

        else:
            # get the solution (variable assignments)
            x_candidate = model.getAttr('X', model.getVars())

            # get the objective value
            x_obj = model.ObjVal

            if x_obj > max_distance and x_obj > best_bound_per_depth[current_node.depth] and found_pos_sol[current_node.depth] == 0 :
                best_bound_per_depth[current_node.depth] = x_obj
                
            if x_obj < max_distance:
                if best_bound_per_depth[current_node.depth] > max_distance:
                    found_pos_sol[current_node.depth] = 1
                    best_bound_per_depth[current_node.depth] = x_obj
                elif x_obj > best_bound_per_depth[current_node.depth]:
                    best_bound_per_depth[current_node.depth] = x_obj
            

            # if we reached the final node of a depth, then update the bounds
            if nodes_per_depth[current_node.depth] == 0:            
                upper_bound = best_bound_per_depth[current_node.depth]
               
        # if infeasible don't create children (continue searching the next node)
        if infeasible:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue

        if abs(lower_bound - upper_bound) < 1e-6: # optimal solution
            
            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
            return solutions, nodes
        
        # check if all variables have integer values (from the ones that are supposed to be integers)
        vars_have_integer_vals = True
        for idx, is_int_var in enumerate(integer_var):
            if is_int_var and idx < len(yi_order) and not is_nearly_integer(x_candidate[yi_order[idx]]):
                vars_have_integer_vals = False
                selected_var_idx = yi_order[idx]
                break
            elif is_int_var and idx > len(yi_order) and not is_nearly_integer(x_candidate[idx]):                   
                vars_have_integer_vals = False
                selected_var_idx = idx
                break

  

        # found feasible solution
        if vars_have_integer_vals: # integer solution
            if lower_bound < x_obj: # a better solution was found
                lower_bound = x_obj
                cost = lower_bound
                if abs(lower_bound - upper_bound) < 1e-6 or abs(lower_bound - max_distance) < 1e-6: # optimal solution
                    # store solution, number of solutions and best sol index (and return)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                        
                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                    return solutions, nodes
                
    

                # Not optimal. Store solution, number of solutions and best sol index (and do not expand children)
                solutions.append([x_candidate, x_obj, current_node.depth])

                # remove the children nodes from each next depth
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 * (i - current_node.depth)

                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                continue

            

            # do not branch further if is an equal solution
            # remove the children nodes from each next depth
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)

            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj,
                            sol_status="Integer (Rejected -- Doesn't improve incumbent)")
            continue

     
        if x_obj < cost or abs(x_obj - cost) < 1e-6: # cut
            # remove the children nodes from each next depth
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)
            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
            continue
 

        if DEBUG_MODE:
            debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional")

        # Retrieve vbasis and cbasis
        vbasis = model.getAttr("VBasis", model.getVars())
        cbasis = model.getAttr("CBasis", model.getConstrs())

        # create lower bounds and upper bounds for child nodes
        left_lb = np.copy(current_node.lb)
        left_ub = np.copy(current_node.ub)
        right_lb = np.copy(current_node.lb)
        right_ub = np.copy(current_node.ub)

        # create left and right branches  (e.g. set left: x = 0, right: x = 1 in a binary problem)
        left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

        # create child nodes
        left_child = Node(left_ub, left_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx,
                          "Left")
        right_child = Node(right_ub, right_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx,
                           "Right")

        # add child nodes in stack  
        stack.append(right_child)
        stack.append(left_child)

    return solutions, nodes


if __name__ == "__main__":

    print("************************    Initializing structures...    ************************")

    P, F, binaryConstraint, fpEucDistances, model, yi_order, max_distance, ub, lb, integer_var, num_vars, c  = pr.pdispersion("test_inputs/25_3_test.txt")

    # Initialize structures
    # A flag for possible objVals that appear for each depth (possible objVals are less orr equal to the max_distance between any two variables)
    found_pos_sol = np.array([0 for i in range(num_vars)])

    # Keep the best bound per depth and the total nodels visited for each depth
    best_bound_per_depth = np.array([-np.inf for i in range(num_vars)])
    
    
    nodes_per_depth = np.zeros(num_vars + 1, dtype=float)
    nodes_per_depth[0] = 1
    for i in range(1, num_vars + 1):
        nodes_per_depth[i] = nodes_per_depth[i - 1] * 2

    # Start solving
    print("************************    Solving problem...    ************************")
    start = time.time()
    solutions, nodes = branch_and_bound(yi_order, P, F, binaryConstraint, fpEucDistances, max_distance, model, ub, lb, integer_var, best_bound_per_depth, found_pos_sol, nodes_per_depth)
    end = time.time()

    if DEBUG_MODE:
        print(f"best_bound_per_depth: {best_bound_per_depth}")
        print(f"nodes_per_depth: {nodes_per_depth}")
    
    if len(solutions) == 0:
        print(f"Infeasible")
    else:
        print(solutions[len(solutions) - 1])
        print(f"Solution Variable Assigment: {solutions[len(solutions) - 1][0]}")
        print(f"Objective value = {solutions[len(solutions) - 1][1]}")

    print(f"Time Elapsed: {end - start}")
    print(f"Total nodes: {nodes}")