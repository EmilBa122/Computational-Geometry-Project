from scipy.optimize import linprog

def B_part_main_function():
    
    c = [3, -12]
    
    constraints = [
        [-1, 2],
        [-2, 3],
        [-1, 3],
        [-1, 6],
        [4, -9]]
    
    values = [
        -1,
        -6,
        0,
        12,
        27]
    
    x0_bounds = (0, None)
    x1_bounds = (0, None)
    
    res = linprog(c, A_ub=constraints, b_ub=values, bounds=[x0_bounds, x1_bounds], method='highs')
    
    x1, x2 = res.x
    max_value = -res.fun
    
    print("------------------------------------------------------------------------------")
    print("OUTPUT of the B part of the project")
    print("X1 solution:", x1)
    print("X2 solution:", x2)
    print("MaxValue :", max_value)
    print("------------------------------------------------------------------------------")



