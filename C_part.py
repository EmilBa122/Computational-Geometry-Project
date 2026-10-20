import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import time

def C_part_main_function(points):
    
    start = time.time()
    
    delaunay = Delaunay(points)
    voronoi = Voronoi(points)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].triplot(points[:, 0], points[:, 1], delaunay.simplices, color='b')
    ax[0].plot(points[:, 0], points[:, 1], 'o', color='r')
    ax[0].set_title('Delaunay')

    voronoi_plot_2d(voronoi, ax=ax[1], show_vertices=False, line_colors='blue', line_width=2, point_size=2)
    ax[1].plot(points[:, 0], points[:, 1], 'o', color='r')
    ax[1].set_title('Voronoi')

    plt.show()
    
    end = time.time()
    
    size_of_dataset = points.shape[0]
    
    print(f"The total time for delaunay triangulation with voronoi diagram is: {end - start}, for "
          + f"{size_of_dataset} number of points")