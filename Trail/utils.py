# utils.py
import numpy as np
from math import pi, hypot

def wrap_angle(angle):
    a = (angle + pi) % (2*pi) - pi
    return a

def euclidean(a,b):
    return hypot(a[0]-b[0], a[1]-b[1])

def to_grid(x, y, resolution):
    gx = int(round(x / resolution))
    gy = int(round(y / resolution))
    return gx, gy

def from_grid(gx, gy, resolution):
    return gx * resolution, gy * resolution

def line_collision(p1, p2, occ_grid, resolution, radius=0.25):
    """
    Sample along straight segment from p1 to p2 and check occupancy around each sample
    using a square neighborhood sufficient to cover the circular radius.
    p1/p2: (x,y) in meters.
    occ_grid: boolean array indexed [row=y_cell, col=x_cell]
    resolution: meters per cell
    radius: robot radius in meters
    """
    dist = euclidean(p1, p2)
    steps = max(int(dist / (resolution * 0.2)), 1)
    h, w = occ_grid.shape
    for i in range(steps+1):
        t = i / steps
        x = p1[0] + (p2[0]-p1[0]) * t
        y = p1[1] + (p2[1]-p1[1]) * t
        gx = int(round(x / resolution))
        gy = int(round(y / resolution))
        # out of bounds -> treat as collision
        if gx < 0 or gy < 0 or gx >= w or gy >= h:
            return True
        rad_cells = int(np.ceil(radius / resolution))
        ymin = max(0, gy - rad_cells)
        ymax = min(h, gy + rad_cells + 1)
        xmin = max(0, gx - rad_cells)
        xmax = min(w, gx + rad_cells + 1)
        if occ_grid[ymin:ymax, xmin:xmax].any():
            return True
    return False
