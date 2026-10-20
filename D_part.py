import numpy as np
import matplotlib.pyplot as plt

# Link of the pseudocode for this kd tree construction
# https://www.baeldung.com/cs/k-d-trees
class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right
       
    def get_point(self):
        return self.point

def kd_tree_constructor(points, k, depth=0):
   if len(points) == 0:
       return None
   
   axis = depth % k
   
   sorted_points = points[points[:, axis].argsort()]
   median_index = len(sorted_points) // 2
   
   return Node(
        sorted_points[median_index],
        left=kd_tree_constructor(sorted_points[:median_index], k, depth + 1),
        right=kd_tree_constructor(sorted_points[median_index + 1:], k, depth + 1))
   
def print_kd_tree(node, position, depth):
    if node is None:
        return

    print("  " * depth + position + ":",  node.get_point())

    if node.left:
        print_kd_tree(node.left, "Left Child" , depth + 1)
    if node.right:
        print_kd_tree(node.right, "Right Child" , depth + 1)

# We see if the region0 contains or it is equal to the region1
def contains_region(region0, region1):
    return all(region0[i][0] <= region1[i][0] and region0[i][1] >= region1[i][1] for i in range(len(region0)))

# We see if the region0 and region1 intersect in any dimension
def intersects_region(region0, region1):
    return not any(region0[i][1] < region1[i][0] or region0[i][0] > region1[i][1] for i in range(len(region0)))

def inside_subtree(node, inside_region_points):
    if node is None: 
        return
   
    inside_subtree(node.left, inside_region_points)
    inside_region_points.append(node.get_point())
    inside_subtree(node.right, inside_region_points)

def range_search(node, current_region, target_region, inside_region_points, k, depth):
    if node is None:
        return

    point = node.get_point()
    axis = depth % k

    # print(f"Depth: {depth}, Axis: {axis}, Point: {point}")

    if all(target_region[i][0] <= point[i] <= target_region[i][1] for i in range(k)):
        inside_region_points.append(point)

    # With this we reduce the current_region every time by the value the coordinate of the variable point
    # has according with the value of axis (if axis = 0 means we refer to the x-axis and if the axis = 1 
    # means we refer to the y-axis)
    left_region = [list(current_region[i]) for i in range(k)]
    # [1] means the upper boundary of the dimension specified by the axis variable
    left_region[axis][1] = point[axis]

    right_region = [list(current_region[i]) for i in range(k)]
    # [0] means the lower boundary of the dimension specified by the axis variable
    right_region[axis][0] = point[axis]

    left_region = tuple(map(tuple, left_region))
    right_region = tuple(map(tuple, right_region))

    if node.left:
        if contains_region(target_region, left_region):
            inside_subtree(node.left, inside_region_points)
        elif intersects_region(target_region, left_region):
            range_search(node.left, left_region, target_region, inside_region_points, k, depth + 1)

    if node.right:
        if contains_region(target_region, right_region):
            inside_subtree(node.right, inside_region_points)
        elif intersects_region(target_region, right_region):
            range_search(node.right, right_region, target_region, inside_region_points, k, depth + 1)

def plot_range_search(all_points, inside_region_points, range_min, range_max):
    range_points = np.array(inside_region_points)
    plt.scatter(all_points[:, 0], all_points[:, 1], c='blue')
    plt.scatter(range_points[:, 0], range_points[:, 1], c='red')
    range_box = plt.Rectangle((range_min[0], range_min[1]), range_max[0] - range_min[0], range_max[1] - range_min[1],
                              edgecolor='green', facecolor='none', linewidth=2, linestyle='--')

    plt.gca().add_patch(range_box)
    plt.show()


def D_part_main_function(points):
    
    k = len(points[0])
    
    kd_tree = kd_tree_constructor(points, k)
    # print_kd_tree(kd_tree, "Root", 0)
    
    range_min = np.array([0.2, 0.1])
    range_max = np.array([0.5, 0.8])
    
    #                          (x_min, x_max)    ,        (y_min, y_max)
    target_region = ((range_min[0], range_max[0]), (range_min[1], range_max[1]))
    k = points.shape[1]
    
    inside_region_points= []
    initialize_region = tuple([(-float('inf'), float('inf'))] * k)
    range_search(kd_tree, initialize_region, target_region, inside_region_points, k, 0)
    
    print("Points within range", range_min, "to", range_max, ":")
    for point in inside_region_points:
        print(point)
    
    plot_range_search(points, inside_region_points, range_min, range_max)