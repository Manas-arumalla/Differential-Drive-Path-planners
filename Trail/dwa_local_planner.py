# dwa_local_planner.py
import numpy as np
from math import hypot, cos, sin

class DWAPlanner:
    def __init__(self, config=None):
        cfg = {
            'v_max': 1.5,
            'v_min': -0.5,
            'omega_max': 1.2,
            'acc_max': 0.8,
            'alpha_max': 1.2,
            'dt': 0.1,
            'predict_time': 2.0,
            'robot_radius': 0.4,
            'to_goal_cost_gain': 1.0,
            'speed_cost_gain': 0.1,
            'obstacle_cost_gain': 1.0
        }
        if config:
            cfg.update(config)
        self.cfg = cfg

    def dynamic_window(self, v, omega):
        cfg = self.cfg
        vs = [max(cfg['v_min'], v - cfg['acc_max']*cfg['dt']),
              min(cfg['v_max'], v + cfg['acc_max']*cfg['dt'])]
        omegas = [max(-cfg['omega_max'], omega - cfg['alpha_max']*cfg['dt']),
                  min(cfg['omega_max'], omega + cfg['alpha_max']*cfg['dt'])]
        return vs, omegas

    def predict_trajectory(self, x, y, theta, v, omega):
        traj = []
        dt = self.cfg['dt']
        t = 0.0
        while t <= self.cfg['predict_time'] + 1e-6:
            x = x + v * cos(theta) * dt
            y = y + v * sin(theta) * dt
            theta = theta + omega * dt
            traj.append((x,y,theta))
            t += dt
        return traj

    def calc_obstacle_cost(self, traj, obstacles):
        minr = float('inf')
        for (tx,ty,_) in traj:
            for (ox,oy,orad) in obstacles:
                d = hypot(tx-ox, ty-oy) - (orad + self.cfg['robot_radius'])
                if d <= 0:
                    return float('inf')
                if d < minr:
                    minr = d
        return 1.0 / minr

    def plan(self, pose, current_vel, current_omega, goal, obstacles):
        vs, omegas = self.dynamic_window(current_vel, current_omega)
        best_score = -float('inf')
        best = (0.0, 0.0)
        for v in np.linspace(vs[0], vs[1], 6):
            for om in np.linspace(omegas[0], omegas[1], 7):
                traj = self.predict_trajectory(pose[0], pose[1], pose[2], v, om)
                last = traj[-1]
                to_goal_cost = -((last[0]-goal[0])**2 + (last[1]-goal[1])**2)**0.5
                speed_cost = v
                obs_cost = -self.calc_obstacle_cost(traj, obstacles)
                score = self.cfg['to_goal_cost_gain']*to_goal_cost + \
                        self.cfg['speed_cost_gain']*speed_cost + \
                        self.cfg['obstacle_cost_gain']*obs_cost
                if score > best_score:
                    best_score = score
                    best = (v, om)
        return best
