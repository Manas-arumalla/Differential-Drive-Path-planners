"""Multi-robot swarm navigation using reciprocal velocity obstacles.

Each robot runs the same :class:`RVOPlanner` in reciprocal (RVO) mode, treating
every other robot as a moving obstacle. The canonical stress test is the
"circle swap": robots on a circle must reach the antipodal point, so every path
crosses the centre at once — collision-free coordination requires reciprocity.
"""
from __future__ import annotations

import numpy as np

from navstack.controllers.velocity_obstacles import RVOPlanner


def simulate_circle_swap(n_robots=8, circle_radius=5.0, robot_radius=0.35,
                         max_speed=1.4, time_horizon=2.5, safety_margin=0.15,
                         dt=0.1, steps=500, goal_tol=0.4):
    """Simulate a reciprocal-RVO circle swap.

    Returns:
        dict with 'trajectories' (list of (T,2) arrays), 'goals' (n,2),
        'min_clearance' (smallest robot-robot gap minus 2*robot_radius seen),
        'all_reached' (bool), 'steps_used' (int).
    """
    angles = np.linspace(0, 2 * np.pi, n_robots, endpoint=False)
    pos = np.array([[circle_radius * np.cos(a), circle_radius * np.sin(a)] for a in angles])
    goals = -pos.copy()  # antipodal
    vel = np.zeros((n_robots, 2))

    planner = RVOPlanner(robot_radius=robot_radius, max_speed=max_speed,
                         time_horizon=time_horizon, reciprocal=True,
                         safety_margin=safety_margin)

    trajs = [[p.copy()] for p in pos]
    min_clear = np.inf
    steps_used = steps
    for k in range(steps):
        new_vel = np.zeros_like(vel)
        for i in range(n_robots):
            if np.hypot(*(goals[i] - pos[i])) < goal_tol:
                new_vel[i] = np.zeros(2)
                continue
            obstacles = [(pos[j, 0], pos[j, 1], vel[j, 0], vel[j, 1], robot_radius)
                         for j in range(n_robots) if j != i]
            new_vel[i] = planner.compute_velocity(pos[i], vel[i], goals[i], obstacles)
        vel = new_vel
        pos = pos + vel * dt
        for i in range(n_robots):
            trajs[i].append(pos[i].copy())
        # track closest robot-robot approach (gap between surfaces)
        for i in range(n_robots):
            for j in range(i + 1, n_robots):
                min_clear = min(min_clear, np.hypot(*(pos[i] - pos[j])) - 2 * robot_radius)
        if all(np.hypot(*(goals[i] - pos[i])) < goal_tol for i in range(n_robots)):
            steps_used = k + 1
            break

    all_reached = all(np.hypot(*(goals[i] - pos[i])) < goal_tol for i in range(n_robots))
    return {
        "trajectories": [np.array(t) for t in trajs],
        "goals": goals,
        "min_clearance": float(min_clear),
        "all_reached": bool(all_reached),
        "steps_used": int(steps_used),
    }
