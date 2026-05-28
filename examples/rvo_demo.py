"""Dynamic obstacle avoidance demo using Velocity Obstacles.

Drives a robot from left to right through a field of crossing obstacles and
renders the trajectory. Saves a figure to media/rvo_demo.png.

    python examples/rvo_demo.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navstack.controllers.velocity_obstacles import RVOPlanner

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")


def run(dt=0.1, steps=320):
    planner = RVOPlanner(robot_radius=0.3, max_speed=1.6, time_horizon=3.0, safety_margin=0.2)
    pos = np.array([0.0, 0.0]); vel = np.array([0.0, 0.0]); goal = np.array([12.0, 0.0])
    # crossing obstacles: (x, y, vx, vy, r)
    obstacles = [
        [4.0, 5.0, 0.0, -0.9, 0.5],
        [7.0, -5.0, 0.0, 1.0, 0.5],
        [9.5, 4.5, -0.2, -0.8, 0.4],
    ]
    traj = [pos.copy()]
    obs_traj = [[o[:2].copy() if hasattr(o, "copy") else list(o[:2])] for o in obstacles]
    for _ in range(steps):
        v = planner.compute_velocity(pos, vel, goal, [tuple(o) for o in obstacles])
        vel = v
        pos = pos + vel * dt
        traj.append(pos.copy())
        for i, o in enumerate(obstacles):
            o[0] += o[2] * dt; o[1] += o[3] * dt
            obs_traj[i].append([o[0], o[1]])
        if np.hypot(*(goal - pos)) < 0.4:
            break
    return np.array(traj), [np.array(t) for t in obs_traj], goal, obstacles


def main():
    traj, obs_traj, goal, obstacles = run()
    os.makedirs(OUT, exist_ok=True)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#1A202C")
    ax.set_facecolor("#1A202C")
    ax.plot(traj[:, 0], traj[:, 1], color="#00D4FF", lw=3, label="robot path", zorder=5)
    ax.plot(traj[0, 0], traj[0, 1], "o", color="#00FF88", ms=14, label="start", zorder=6)
    ax.plot(goal[0], goal[1], "*", color="#FF4444", ms=22, label="goal", zorder=6)
    for i, ot in enumerate(obs_traj):
        ax.plot(ot[:, 0], ot[:, 1], "--", color="#FFA500", lw=1.2, alpha=0.7)
        ax.add_patch(plt.Circle(ot[-1], obstacles[i][4], color="#FF6B6B", alpha=0.6))
    ax.set_title("Dynamic Obstacle Avoidance (Velocity Obstacles)", color="white", fontweight="bold")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
    ax.legend(loc="upper right", facecolor="#2D3748", labelcolor="white")
    out = os.path.join(OUT, "rvo_demo.png")
    fig.savefig(out, dpi=150, facecolor="#1A202C", bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
