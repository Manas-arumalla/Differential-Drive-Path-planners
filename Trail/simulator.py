# simulator.py
import numpy as np
from vehicle import KinematicBicycle
from matplotlib.patches import Circle

class DynamicObstacle:
    def __init__(self, x,y,vx,vy,r=0.3):
        self.x = float(x); self.y = float(y)
        self.vx = float(vx); self.vy = float(vy)
        self.r = float(r)

    def step(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

    def as_tuple(self):
        return (self.x, self.y, self.r)

class Simulator:
    def __init__(self, occ_grid, resolution=0.5, dt=0.1):
        self.occ = occ_grid
        self.res = resolution
        self.dt = dt
        self.veh = None
        self.dynamic_obstacles = []

    def set_vehicle(self, veh):
        self.veh = veh

    def add_dynamic_obstacle(self, obs):
        self.dynamic_obstacles.append(obs)

    def step(self, v, delta):
        if self.veh is None:
            raise RuntimeError("Vehicle not set")
        self.veh.step(v, delta)
        for o in self.dynamic_obstacles:
            o.step(self.dt)

    def get_obstacles(self):
        return [o.as_tuple() for o in self.dynamic_obstacles]

    def plot(self, ax, start, goal, path=None, smoothed=None):
        ax.clear()
        h, w = self.occ.shape
        extent = [0, w*self.res, 0, h*self.res]
        ax.imshow(self.occ, origin='lower', extent=extent, cmap='gray_r')
        for o in self.dynamic_obstacles:
            c = Circle((o.x, o.y), o.r, color='r', alpha=0.8)
            ax.add_patch(c)
        if self.veh:
            c2 = Circle((self.veh.x, self.veh.y), 0.3, color='b', alpha=0.8)
            ax.add_patch(c2)
            ax.arrow(self.veh.x, self.veh.y, 0.6*np.cos(self.veh.theta), 0.6*np.sin(self.veh.theta),
                     head_width=0.1, color='b')
        ax.plot(start[0], start[1], 'go', label='start')
        ax.plot(goal[0], goal[1], 'mx', label='goal')
        if path is not None:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, '-y', linewidth=1, label='global')
        if smoothed is not None:
            xs = [p[0] for p in smoothed]
            ys = [p[1] for p in smoothed]
            ax.plot(xs, ys, '-c', linewidth=2, label='smoothed')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.set_title('Hybrid A* (hierarchical) + DWA Simulation')
