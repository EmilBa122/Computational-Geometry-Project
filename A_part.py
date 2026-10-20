import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import random
import math
import matplotlib.animation as animation
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from functools import cmp_to_key

# -------------------------------Common functions part------------------------------------#
def orientation(p0, p1, p2):
    """
    Determine the orientation of the triplet (p0, p1, p2)
    
    Returns:
        
    - 0 if the points are collinear
    - 1 if the points are in clockwise (CW) order
    - 2 if the points are in counterclockwise (CCW) order
    """
    # Calculate the cross product of vectors p1p0 and p2p0
    cross_product = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])

    if cross_product == 0:
        return 0  # Collinear
    elif cross_product > 0:
        return 1  # Clockwise
    else:
        return 2  # Counterclockwise
#---------------------------------------------------------------------------------# 

#---------------------------Graham Scan part--------------------------------------#
def points_above_below_line(points, a, b):
    P_above = np.vstack([a])
    P_below = np.vstack([a])
    
    for p in points:
        if np.array_equal(p, a) or np.array_equal(p, b):
            continue
        elif np.cross(p-a, b-a) <= 0:
            # is True the point(s) lie above the line
            P_above = np.vstack([P_above, p])  
        else:
            P_below = np.vstack([P_below, p])
    
    P_above = np.vstack([P_above, b])  
    P_below = np.vstack([P_below, b])
    
    return P_above, P_below   


def incremental_Graham_Scan(points):
    # Sort points lexicographically (by x, then by y)
    sorted_points = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    P_above, P_below = points_above_below_line(sorted_points, sorted_points[0], sorted_points[-1])
    
    # Build upper hull
    L_upper = np.vstack([P_above[0], P_above[1]])
   
    for i in range(2, len(P_above)):
        while (len(L_upper) > 2):
            if (orientation(L_upper[-1], L_upper[-2], L_upper[-3]) != 1):
                L_upper = np.delete(L_upper, -2, axis=0)
            else:
                break
        L_upper = np.vstack([L_upper, P_above[i]])  
    
    # Build lower hull
    L_lower = np.vstack([P_below[-1], P_below[-2]])
    
    for i in range(len(P_below) - 3, -1, -1):
        while (len(L_lower) > 2):
            if (orientation(L_lower[-1], L_lower[-2], L_lower[-3]) == 2):
                L_lower = np.delete(L_lower, -2, axis=0)
            else:
                break
        L_lower = np.vstack([L_lower, P_below[i]]) 
            
    L_lower = np.delete(L_lower, 0, axis=0)
    L_lower = np.delete(L_lower, -1, axis=0)   
    
    L = np.vstack([L_upper, L_lower]) 
    
    # print(L)
    
    return L  


def Graham_Scan_implementation(points):
    hull = incremental_Graham_Scan(points)
    
    x = points[:,0]
    y = points[:,1]
    plt.plot(x, y, 'bo')
    n = len(hull)
    for i in range(n):
        # current point on the hull
        current_point = hull[i]
        # next point on the hull, using modulo to wrap around to the first point when reaching the end of the array
        next_point = hull[(i + 1) % len(hull)]
        
        plt.plot([current_point[0], next_point[0]], [current_point[1], next_point[1]], 'ro', linestyle="--")
    plt.title("Graham Scan algorithm")
    plt.show()
    
    return hull
#---------------------------------------------------------------------------------#   

#---------------------------Gift Wrapping part------------------------------------#
def gift_wrapping(points):   
    S = points
    n = len(points)
    hull = [] 
    # Find the leftmost point
    index_r0 = np.argmin(S[:, 0])           
    index_r = index_r0
    
    while True:
        
        hull.append(S[index_r])
        
        index_u = (index_r + 1) % n
        
        for index_t in range(n):
            result = orientation(S[index_r], S[index_u], S[index_t])
            if ((result == 1) or ((result == 0) and (S[index_r][0] < S[index_u][0] < S[index_t][0]) and
                                  (S[index_r][1] < S[index_u][1] < S[index_t][1]))):
                index_u = index_t
        if index_u == index_r0:
            break
        
        index_r = index_u
        
        
    return np.array(hull)    


def Gift_Wrapping_implementation(points):
    hull = gift_wrapping(points)
    
    x = points[:,0]
    y = points[:,1]
    plt.plot(x, y, 'bo')
    for i in range(len(hull)):
            # current point on the hull
            current_point = hull[i]
            # next point on the hull, using modulo to wrap around to the first point when reaching the end of the array
            next_point = hull[(i + 1) % len(hull)]
            plt.plot([current_point[0], next_point[0]], [current_point[1], next_point[1]], 'ro', linestyle="--")
    plt.title("Gift Wrapping algorithm")
    plt.show()
    
    return hull
#-------------------------------------------------------------------------------------#

#---------------------------Divide and Conquer part-----------------------------------#    
# Code for the Divide and Conquer algorithm from
# https://www.geeksforgeeks.org/convex-hull-using-divide-and-conquer-algorithm/

def quad(p):
    if p[0] >= 0 and p[1] >= 0:
        return 1
    if p[0] <= 0 and p[1] >= 0:
        return 2
    if p[0] <= 0 and p[1] <= 0:
        return 3
    return 4


# Checks whether the line is crossing the polygon
def orientation_divide_conquer(a, b, c):
    res = (b[1]-a[1]) * (c[0]-b[0]) - (c[1]-b[1]) * (b[0]-a[0])
    if res == 0:
        return 0
    if res > 0:
        return 1
    return -1


# compare function for sorting
def compare(p1, q1):
    p = [p1[0]-mid[0], p1[1]-mid[1]]
    q = [q1[0]-mid[0], q1[1]-mid[1]]
    one = quad(p)
    two = quad(q)
    if one != two:
        if one < two:
            return -1
        return 1
    if p[1]*q[0] < q[1]*p[0]:
        return -1
    return 1


# Finds upper tangent of two polygons 'a' and 'b'
def merger(a, b):
    # n1 -> number of points in polygon a
    # n2 -> number of points in polygon b
    n1, n2 = len(a), len(b)
    ia, ib = 0, 0
    # ia -> rightmost point of a
    for i in range(1, n1):
        if a[i][0] > a[ia][0]:
            ia = i
    # ib -> leftmost point of b        
    for i in range(1, n2):
        if b[i][0] < b[ib][0]:
            ib = i
    # finding the upper tangent        
    inda, indb = ia, ib
    done = 0
    while not done:
        done = 1
        while orientation_divide_conquer(b[indb], a[inda], a[(inda+1) % n1]) >= 0:
            inda = (inda + 1) % n1
        while orientation_divide_conquer(a[inda], b[indb], b[(n2+indb-1) % n2]) <= 0:
            indb = (indb - 1) % n2
            done = 0
    uppera, upperb = inda, indb
    inda, indb = ia, ib
    done = 0
    while not done:     # finding the lower tangent
        done = 1
        while orientation_divide_conquer(a[inda], b[indb], b[(indb+1) % n2]) >= 0:
            indb = (indb + 1) % n2
        while orientation_divide_conquer(b[indb], a[inda], a[(n1+inda-1) % n1]) <= 0:
            inda = (inda - 1) % n1
            done = 0
    ret = []
    lowera, lowerb = inda, indb
    # ret contains the convex hull after merging the two convex hulls
    # with the points sorted in anti-clockwise order
    ind = uppera
    ret.append(a[uppera])
    while ind != lowera:
        ind = (ind+1) % n1
        ret.append(a[ind])
    ind = lowerb
    ret.append(b[lowerb])
    while ind != upperb:
        ind = (ind+1) % n2
        ret.append(b[ind])
    return ret


# Brute force algorithm to find convex hull for a set 
# of less than 6 points
def bruteHull(a):
    # Take any pair of points from the set and check
    # whether it is the edge of the convex hull or not.
    # if all the remaining points are on the same side
    # of the line then the line is the edge of convex
    # hull otherwise not
    global mid
    s = set()
    for i in range(len(a)):
        for j in range(i+1, len(a)):
            x1, x2 = a[i][0], a[j][0]
            y1, y2 = a[i][1], a[j][1]
            a1, b1, c1 = y1-y2, x2-x1, x1*y2-y1*x2
            pos, neg = 0, 0
            for k in range(len(a)):
                if (k == i) or (k == j) or (a1*a[k][0]+b1*a[k][1]+c1 <= 0):
                    neg += 1
                if (k == i) or (k == j) or (a1*a[k][0]+b1*a[k][1]+c1 >= 0):
                    pos += 1
            if pos == len(a) or neg == len(a):
                s.add(tuple(a[i]))
                s.add(tuple(a[j]))
    ret = []
    for x in s:
        ret.append(list(x))
    # Sorting the points in the anti-clockwise order
    mid = [0, 0]
    n = len(ret)
    for i in range(n):
        mid[0] += ret[i][0]
        mid[1] += ret[i][1]
        ret[i][0] *= n
        ret[i][1] *= n
    ret = sorted(ret, key=cmp_to_key(compare))
    for i in range(n):
        ret[i] = [ret[i][0]/n, ret[i][1]/n]
    return ret


# Returns the convex hull for the given set of points
def divide(points):
    # If the number of points is less than 6 then the
    # function uses the brute algorithm to find the
    # convex hull
    if len(points) <= 5:
        return bruteHull(points)
    # left contains the left half points
    # right contains the right half points
    left, right = [], []
    start = int(len(points)/2)
    for i in range(start):
        left.append(points[i])
        
    for i in range(start, len(points)):
        right.append(points[i])
    
    # convex hull for the left and right sets
    left_hull = divide(left)
    right_hull = divide(right)
    # merging the convex hulls
    return merger(left_hull, right_hull)


def Divide_and_Conquer_implementation(points):
    temp_points = sorted(points.tolist())
    ans = divide(temp_points)
    # print(ans)
    # print('Convex Hull:')
    # for x in ans:
    #     print(x[0], x[1])
        
    # Plotting the points and the convex hull
    points_x, points_y = zip(*temp_points)
    hull_x, hull_y = zip(*ans)
    
    plt.plot(points_x, points_y, 'bo')
    plt.plot(hull_x + (hull_x[0],), hull_y + (hull_y[0],), 'ro', linestyle="--")
    plt.title('Divide and Conquer algorithm')
    plt.show()    
#-------------------------------------------------------------------------------------#

#-------------------------------Quick Hull part---------------------------------------#
# Code on how to find a point is above or below a line
# https://stackoverflow.com/questions/62881659/how-to-find-a-point-is-on-left-or-right-side-of-line-using-python-and-dlib
# https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
def point_side_identification(p0, p1, p2):
    return (p2[1] - p0[1]) * (p1[0] - p0[0]) - (p1[1] - p0[1]) * (p2[0] - p0[0])

def distance(p0, p1, p2):
    return abs(point_side_identification(p0, p1, p2))

def hull(p0, p1, side, points, hull_points):
    ind = -1
    max_dist = 0

    for i in range(len(points)):
        temp = distance(p0, p1, points[i])
        if ((point_side_identification(p0, p1, points[i]) * side) > 0) and temp > max_dist:
            ind = i
            max_dist = temp

    if ind == -1:
        hull_points.add(tuple(p0))
        hull_points.add(tuple(p1))
        return

    hull(points[ind], p0, -point_side_identification(points[ind], p0, p1), points, hull_points)
    hull(points[ind], p1, -point_side_identification(points[ind], p1, p0), points, hull_points)

def quick_hull(points):
   
    hull_points = set()
    
    # Extract x coordinates (first column)
    x_coordinate = points[:, 0]
   
    # We have the indexes of points with the maximum/minimum x coordinate
    min_index = np.argmin(x_coordinate)
    max_index = np.argmax(x_coordinate)    

    min_x_point = points[min_index]
    max_x_point = points[max_index]

    hull(min_x_point, max_x_point, 1, points, hull_points)
    hull(min_x_point, max_x_point, -1, points, hull_points)

    hull_list = list(hull_points)
    
    # print(hull_list)
    
    centroid = np.mean(hull_list, axis=0)

    # Links on how to sort points in counterclockwise order using centroid
    # https://blogs.sas.com/content/iml/2021/11/17/order-vertices-convex-polygon.html
    # https://math.stackexchange.com/questions/1329128/how-to-sort-vertices-of-a-polygon-in-counter-clockwise-order-computing-angle
    hull_list.sort(key=lambda p: math.atan2(p[1] - centroid[1], p[0] - centroid[0]))

    return np.array(hull_list)

def Quick_Hull_implementation(points):

    hull_points = quick_hull(points)
  
    x = points[:,0]
    y = points[:,1]
    plt.plot(x, y, 'bo') 
    for i in range(len(hull_points)):
        current_point = hull_points[i]
        # next point on the hull, using modulo to wrap around to the first point when reaching the end of the array
        next_point = hull_points[(i + 1) % len(hull_points)]
        plt.plot([current_point[0], next_point[0]], [current_point[1], next_point[1]], 'ro', linestyle="--")
    plt.title('Quick Hull algorithm')
    plt.show()
#-------------------------------------------------------------------------------------#

#-------------------------A3 Question (collinear points) part-------------------------#
def create_smallest_dataset(points, number_of_points):
    
    size_of_dataset = points.shape[0]
    
    print(size_of_dataset)
    # The new smallest dataset
    new_points =  np.array([points[np.random.randint(0, size_of_dataset - 1)]])
    
    i = 1
    while i < number_of_points:
        
        random_point = points[random.randint(0, size_of_dataset - 1)]
        
        # Check if the point exist already in the ndarray
        is_inside = np.any(np.all(new_points == random_point, axis=1))
        if is_inside:
            continue
        new_points = np.vstack([new_points, random_point])
        i += 1
    
    return new_points    


# Function to generate n collinear points for a given dataset
def generate_n_collinear_points_for_given_dataset(points, n):
    size_of_dataset = points.shape[0]
    print(size_of_dataset)
    i = 0
    while i < n:
        collinear_point = []
        # Select two random points from the dataset
        p0 = points[random.randint(0, size_of_dataset - 1)]
        p1 = points[random.randint(0, size_of_dataset - 1)]
        if not np.array_equal(p0, p1):
            
            # Compute line equation ax + by + c = 0
            a = p1[1] - p0[1]
            b = p0[0] - p1[0]
            c = p1[0] * p0[1] - p0[0] * p1[1]
            
            # Generate parameter t between 0 and 1
            t = random.uniform(0, 1)
            # Interpolate between p0 and p1
            x = p0[0] * (1 - t) + p1[0] * t
            y = p0[1] * (1 - t) + p1[1] * t
            
            collinear_point.append([x, y])
            
            # Convert list of collinear points to numpy array
            collinear_point = np.array(collinear_point)
            
            points = np.vstack([points, collinear_point])
            
            i += 1
    return points   
#-------------------------------------------------------------------------------------#

#------------------------A4 Question (visualization of the steps) part----------------#
def gift_wrapping_by_steps(points):
    S = points
    n = len(points)
    hull = []
    index_r0 = np.argmin(S[:, 0])          
    index_r = index_r0
    step_plots = []
   
    while True:
        hull.append(S[index_r])
        step_plots.append(np.array(hull))
        index_u = (index_r + 1) % n
        for index_t in range(n):
            result = orientation(S[index_r], S[index_u], S[index_t])
            if ((result == 1) or ((result == 0) and (S[index_r][0] < S[index_u][0] < S[index_t][0]) and
                                  (S[index_r][1] < S[index_u][1] < S[index_t][1]))):
                index_u = index_t
        if index_u == index_r0:
            hull.append(S[index_r0])  # Close the hull
            step_plots.append(np.array(hull))
            break
        index_r = index_u

    return step_plots


def Gift_Wrapping_by_steps_implementation(points):
    steps = gift_wrapping_by_steps(points)
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
   
    # Function to update the plot
    def update(num):
        ax.clear()
        ax.scatter(points[:, 0], points[:, 1], label='Points')
        hull_points = steps[num]
        ax.plot(hull_points[:, 0], hull_points[:, 1], 'r-', label='Current Hull')
        ax.scatter(hull_points[:, 0], hull_points[:, 1], c='red')
        ax.set_title(f'Step {num + 1}')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
       
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(steps), repeat=False)
   
    # Save the animation as a GIF file
    gif_path = 'giftwrap.gif'
    ani.save(gif_path, writer='pillow')
    print(f"GIF saved at: {os.path.abspath(gif_path)}")
   
    # Display the animation
    plt.legend()
    plt.show()
#----------------------------------------------------------------------------------#

#------------------------------Convex Hull 3d--------------------------------------#
def convex_hull_3d(points):  
    hull = ConvexHull(points)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')

    for simplex in hull.simplices:
        poly = Poly3DCollection([points[simplex]], alpha=0.5, linewidths=1, edgecolors='r')
        poly.set_facecolor('cyan')
        ax.add_collection3d(poly)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("3D Convex hull points")
    print(points[hull.vertices])
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
#------------------------Initialization part function/
#                        Access A part algorithms function------------------------#
# Generate n random points
def generate_n_random_points(n, dimension):
    points = [[random.uniform(0, 1) for i in range(dimension)] for _ in range(n)]
    return np.array(points)


def access_A_part_algorithms(points):
    start = time.time()
    hull_p = Graham_Scan_implementation(points)
    end = time.time()
    print(f"With total time: {end - start} for the Graham_Scan algorithm")
    start = time.time()
    hull_points = Gift_Wrapping_implementation(points)
    end = time.time()
    print(f"With total time: {end - start} for the Gift_Wrapping algorithm")
    start = time.time()
    Divide_and_Conquer_implementation(points)
    end = time.time()
    print(f"With total time: {end - start} for the Divide_and_Conquer algorithm")
    start = time.time()
    Quick_Hull_implementation(points)
    end = time.time()
    print(f"With total time: {end - start} for the Quick_Hull algorithm")
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Convex hull points from Graham Scan")
    print(hull_p)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("Convex hull points from all the other algorithms")
    print(hull_points)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#-------------------------------------------------------------------------------------# 