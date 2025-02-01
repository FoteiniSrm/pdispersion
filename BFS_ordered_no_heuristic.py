from gurobipy import GRB
import numpy as np
import time
import problems as pr
import heuristics as heur
from priority_queue import PriorityQueue

# define global variables
DEBUG_MODE = False # debug enabled / disabled
lower_bound = -np.inf # lower bound of the problem
upper_bound = np.inf # upper bound of the problems

# calculate if a value is very close to an integer value
def is_nearly_integer(value, tolerance=1e-6):
    return abs(value - round(value)) <= tolerance

# a class 'Node' that holds information of a node
class Node:
    def __init__(self, ub, lb, depth, vbasis, cbasis, branching_var, objValue, x_candidate, label=""):
        self.ub = ub
        self.lb = lb
        self.depth = depth
        self.vbasis = vbasis
        self.cbasis = cbasis
        self.branching_var = branching_var
        self.objValue = objValue
        self.x_candidate = x_candidate
        self.label = label

    def __lt__(self, other):
        # Compare Nodes based on objValue 
        return self.objValue < other.objValue
    
    def __repr__(self):
        return f"Node(branching_var={self.branching_var}, objValue={self.objValue}, label={self.label},)"

def process_node(model, lb, ub, vbasis, cbasis, cost, depth, nodes_per_depth, pq, parent_node, selected_var_idx, label, is_left=True):
    """
    Helper function to process a node (left or right) and create child nodes if feasible.
    """
    global DEBUG_MODE

    # Warm start solver
    if vbasis and cbasis:
        model.setAttr("VBasis", model.getVars(), vbasis)
        model.setAttr("CBasis", model.getConstrs(), cbasis)

    # update the state of the model, passing the new lower bounds/upper bounds for the vars.
    # Basically, we only change the ub/lb for the branching variable. Another way is to introduce a new constraint (e.g. x_i <= ub).
    model.setAttr("LB", model.getVars(), lb)
    model.setAttr("UB", model.getVars(), ub)
    model.update()

    # Optimize the model
    model.optimize()

    # Check if the model was solved to optimality
    if model.status != GRB.OPTIMAL:
        # Cut all future node children from the tree
        for i in range(depth + 1, len(nodes_per_depth)):
            nodes_per_depth[i] -= 2 * (i - depth)
        if DEBUG_MODE:
            print("Infeasible")
        return

    # Get the solution and objective value
    obj_val = model.ObjVal
    if obj_val < cost and abs(obj_val - cost) >= 1e-6:
        # Remove the children nodes from each next depth
        for i in range(parent_node.depth + 1, len(nodes_per_depth)):
            nodes_per_depth[i] -= 2 * (i - parent_node.depth)
        if DEBUG_MODE:
            debug_print(node=parent_node, x_obj=obj_val, sol_status="Fractional -- Cut by bound")
        return

    # Get the solution (variable assignments)
    x_candidate = model.getAttr('X', model.getVars())

    # Create child node
    child_lb = np.copy(lb)
    child_ub = np.copy(ub)
    child_vbasis = model.getAttr("VBasis", model.getVars())
    child_cbasis = model.getAttr("CBasis", model.getConstrs())

    child_node = Node(
        child_ub, child_lb, parent_node.depth + 1, child_vbasis, child_cbasis,
        selected_var_idx, obj_val, x_candidate, label
    )

    if DEBUG_MODE:
        debug_print(node=child_node, x_obj=obj_val, sol_status="Created")

    # Add child node to the priority queue
    pq.push(child_node)


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
    if node is not None:
        print(f"ObjVal: {node.objValue}")  
    if x_obj is not None:
        print(f"Simplex Objective: {x_obj}")
    if sol_status is not None:
        print(f"Solution status: {sol_status}")

    # print("\n\n--------------------------------------------------\n\n")


# branch & bound algorithm
def branch_and_bound(yi_order, P, F, binaryConstraint, fpEucDistances, max_distance, model, ub, lb, integer_var, best_bound_per_depth, found_pos_sol, nodes_per_depth, vbasis=[], cbasis=[], depth=0):
    global lower_bound, upper_bound

    # Create a priority queue to perform a Best First Search based on the objective value calculated for each node
    pq = PriorityQueue()

    # initialize solution list
    solutions = list()
    nodes = 0

    
    # node at root will be explored
    nodes_per_depth[0] -= 1  

    # solve relaxed problem
    model.setParam('OutputFlag', 0) #supress gurobi output
    model.optimize()

    # check if the model was solved to optimality. If not then return (infeasible).
    if model.status != GRB.OPTIMAL:  
        if DEBUG_MODE:    
            print("Problem was determined to be infeasible")
        return [], nodes
        

    # get the solution (variable assignments)
    x_candidate = model.getAttr('X', model.getVars())

    # get the objective value
    x_obj = model.ObjVal
    best_bound_per_depth[0] = x_obj

    # ===============  Root node  ==========================
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, x_obj, x_candidate, "root")
    if DEBUG_MODE:
        debug_print()

    nodes += 1

    # check if all variables have integer values (from the ones that are supposed to be integers).
    # If not, then select the first in order variable with a fractional value to be the one fixed.
    # If the non integer is a decision variable pick the most appopriate one fom the yi_order
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
            print("Found optimal solution at root")
        return solutions, nodes

    # otherwise update upper bound (Maximization)
    else:
        upper_bound = x_obj       

    if DEBUG_MODE:
        debug_print(node=root_node, x_obj=x_obj, sol_status="Fractional")

    # # Call the heuristic function
    # status, solution, cost = heur.random_multiple_runs_heuristic(10, 10000, P, F, fpEucDistances, binaryConstraint)
    cost = -np.inf

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

    # Expand the nodes and calculate objVals. That way the node with the largest objVals will get checked first (Best First Search)
    # The lb, ub, vbasis, cbasis entered in the node will be used to calculate the node after the one we just expanded (if needed)

    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  LEFT NODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    process_node(
        model, left_lb, left_ub, vbasis, cbasis, cost, depth, nodes_per_depth,
        pq, root_node, selected_var_idx, "Left", is_left=True
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  RIGHT NODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    process_node(
        model, right_lb, right_ub, vbasis, cbasis, cost, depth, nodes_per_depth,
        pq, root_node, selected_var_idx, "Right", is_left=False
    )

    

    # solving sub problems
    # While the pq has nodes, continue solving
    while not pq.is_empty():
        
        if DEBUG_MODE:
            print("\n********************************  NEW NODE BEING EXPLORED  ******************************** ")

        # increment total nodes by 1
        nodes += 1

        # get the child node with largest objVal from priority queue
        current_node = pq.pop()

        # increase the nodes visited for current depth
        nodes_per_depth[current_node.depth] -= 1
  
        
        # get the solution (variable assignments)
        x_candidate = current_node.x_candidate

        # get the objective value
        x_obj = current_node.objValue

        
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

        if x_obj > max_distance and x_obj > best_bound_per_depth[current_node.depth] and found_pos_sol[current_node.depth] == 0 :
            best_bound_per_depth[current_node.depth] = x_obj
            
        if x_obj <= max_distance and x_obj >= lower_bound and vars_have_integer_vals:
            if best_bound_per_depth[current_node.depth] >= max_distance:
                found_pos_sol[current_node.depth] = 1
                best_bound_per_depth[current_node.depth] = x_obj
            elif x_obj >= best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj
        

        # if we reached the final node of a depth, then update the UB (since its max problem)
        if nodes_per_depth[current_node.depth] == 0:
            upper_bound = best_bound_per_depth[current_node.depth]


        if abs(lower_bound - upper_bound) < 1e-6: # optimal solution
    
            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")

            return solutions, nodes



        # found feasible solution
        if vars_have_integer_vals: # integer solution
     
            if lower_bound < x_obj: # a better integer solution was found - update LB (since its max problem)
                lower_bound = x_obj
                cost = lower_bound
                if abs(lower_bound - upper_bound) < 1e-6 or abs(lower_bound - max_distance) < 1e-6: # optimal solution (If LB = UB or if lower bound is equal to the max possible distance between two sites)
                    # store optimal solution and return
                    solutions.append([x_candidate, x_obj, current_node.depth])

                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")

                    return solutions, nodes
                
    
                # Not optimal. Store solution and do not expand children
                solutions.append([x_candidate, x_obj, current_node.depth])

                # remove the children nodes from each next depth
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 * (i - current_node.depth)

                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                continue


            
            # If we have a feasible integer solution that is equal or smaller than the LB do not branch further (max problem)
            # remove the children nodes from each next depth
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)

            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj,
                            sol_status="Integer (Rejected -- Doesn't improve incumbent)")
            continue

        
        # If we have a fractional solution that is smaller or equal to the LB do not branch further (max problem)
        if x_obj < lower_bound or abs(x_obj - lower_bound) < 1e-6: # cut
            # remove the children nodes from each next depth
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)
            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
            continue
        

        # If we have a fractional solution we branch further 
        if DEBUG_MODE:
            debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional")

        left_lb = np.copy(current_node.lb)
        left_ub = np.copy(current_node.ub)
        right_lb = np.copy(current_node.lb)
        right_ub = np.copy(current_node.ub)

        # create left and right branches (e.g. set left: x = 0, right: x = 1 in a binary problem)
        left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])
            
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  LEFT NODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        process_node(
            model, left_lb, left_ub, current_node.vbasis, current_node.cbasis, cost, current_node.depth, nodes_per_depth,
            pq, current_node, selected_var_idx, "Left", is_left=True
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  RIGHT NODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        process_node(
            model, right_lb, right_ub, current_node.vbasis, current_node.cbasis, cost, current_node.depth, nodes_per_depth,
            pq, current_node, selected_var_idx, "Right", is_left=False
        )
        
    return solutions, nodes


if __name__ == "__main__":

    print("************************    Initializing structures...    ************************")

    P, F, binaryConstraint, fpEucDistances, model, yi_order, max_distance, ub, lb, integer_var, num_vars, c = pr.pdispersion("25_3_test.txt")

    # Initialize structures
    # A flag for possible objVals that appear for each depth (possible objVals are less orr equal to the max_distance between any two variables)
    found_pos_sol = np.array([0 for i in range(num_vars)])

    # Keep the best bound per depth and the total nodels visited for each depth (for max problem set all to -inf)  
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

