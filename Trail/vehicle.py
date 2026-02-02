# vehicle.py
import numpy as np
from math import cos, sin, tan
import numpy as _np

class KinematicBicycle:
    def __init__(self, x=0, y=0, theta=0, L=0.5, dt=0.1, wheelbase=None, max_steer=0.6):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
        self.dt = float(dt)
        self.L = float(wheelbase) if wheelbase is not None else float(L)
        self.max_steer = float(max_steer)
        self.v = 0.0
        self.omega = 0.0

    def step(self, v, delta):
        """
        Update pose using simple kinematic bicycle forward Euler integration.
        v: forward speed (m/s)
        delta: steering angle (rad)
        """
        delta = max(-self.max_steer, min(self.max_steer, delta))
        self.x += v * cos(self.theta) * self.dt
        self.y += v * sin(self.theta) * self.dt
        # avoid division by zero issues; small v still gives finite omega
        self.theta += (v / self.L) * tan(delta) * self.dt
        self.theta = (self.theta + _np.pi) % (2*_np.pi) - _np.pi
        self.v = v
        self.omega = (v / self.L) * tan(delta)
        return self.x, self.y, self.theta
