"""Generate cinematic GIFs for the README / portfolio.

    python examples/make_gifs.py            # all GIFs
    python examples/make_gifs.py --rvo      # just the RVO clip
    python examples/make_gifs.py --astar    # just the A* search clip

Outputs land in media/.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from navstack.environment import create_demo_environment
from navstack.planners.astar_planner import AStarPlanner
from navstack.controllers.velocity_obstacles import RVOPlanner

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")
plt.style.use("dark_background")


def make_rvo_gif(path, dt=0.1, steps=300, fps=25):
    planner = RVOPlanner(robot_radius=0.3, max_speed=1.6, time_horizon=3.0, safety_margin=0.2)
    pos = np.array([0.0, 0.0]); vel = np.array([0.0, 0.0]); goal = np.array([12.0, 0.0])
    obstacles = [[4.0, 5.0, 0.0, -0.9, 0.5], [7.0, -5.0, 0.0, 1.0, 0.5], [9.5, 4.5, -0.2, -0.8, 0.4]]
    frames = []
    for _ in range(steps):
        v = planner.compute_velocity(pos, vel, goal, [tuple(o) for o in obstacles])
        vel = v; pos = pos + vel * dt
        for o in obstacles:
            o[0] += o[2] * dt; o[1] += o[3] * dt
        frames.append((pos.copy(), [(o[0], o[1], o[4]) for o in obstacles]))
        if np.hypot(*(goal - pos)) < 0.4:
            break

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#1A202C")
    trail = []

    def draw(i):
        ax.clear(); ax.set_facecolor("#1A202C")
        p, obs = frames[i]; trail.append(p)
        tr = np.array(trail)
        ax.plot(tr[:, 0], tr[:, 1], color="#00D4FF", lw=2.5, alpha=0.9)
        ax.plot(p[0], p[1], "o", color="#00D4FF", ms=12, mec="white")
        ax.plot(0, 0, "o", color="#00FF88", ms=10); ax.plot(goal[0], goal[1], "*", color="#FF4444", ms=20)
        for ox, oy, orr in obs:
            ax.add_patch(plt.Circle((ox, oy), orr, color="#FF6B6B", alpha=0.7))
        ax.set_xlim(-1, 13); ax.set_ylim(-6, 6); ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        ax.set_title("Dynamic Obstacle Avoidance — Velocity Obstacles", color="white", fontweight="bold")

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"Saved {path} ({len(frames)} frames)")


def make_astar_gif(path, fps=20):
    env = create_demo_environment()
    start, goal = (3.0, 3.0), (45.0, 45.0)
    planner = AStarPlanner(env, robot_radius=0.5)
    result = planner.plan(start, goal)
    explored = planner.get_explored_nodes()

    n_reveal = 40
    chunk = max(1, len(explored) // n_reveal)
    reveal_frames = list(range(0, len(explored), chunk))
    hold = 15  # frames to hold the final path

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#1A202C")

    def draw(fi):
        ax.clear(); ax.set_facecolor("#1A202C")
        ax.imshow(env.grid, origin="lower", extent=[0, env.width, 0, env.height],
                  cmap="bone_r", alpha=0.85)
        if fi < len(reveal_frames):
            k = reveal_frames[fi]
            exp = np.array(explored[:k + chunk]) if explored else np.empty((0, 2))
            if len(exp):
                ax.scatter(exp[:, 0], exp[:, 1], c=np.arange(len(exp)), cmap="cool", s=6, alpha=0.5)
        else:
            exp = np.array(explored)
            if len(exp):
                ax.scatter(exp[:, 0], exp[:, 1], c="#0088AA", s=5, alpha=0.3)
            if result:
                pr = np.array(result)
                ax.plot(pr[:, 0], pr[:, 1], color="#00D4FF", lw=3.5)
        ax.plot(start[0], start[1], "o", color="#00FF88", ms=12)
        ax.plot(goal[0], goal[1], "*", color="#FF4444", ms=20)
        ax.set_xlim(0, env.width); ax.set_ylim(0, env.height); ax.set_aspect("equal")
        ax.set_title("A* Search — exploration then optimal path", color="white", fontweight="bold")

    total = len(reveal_frames) + hold
    anim = FuncAnimation(fig, draw, frames=total, interval=1000 / fps)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"Saved {path} ({total} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvo", action="store_true")
    ap.add_argument("--astar", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    do_all = not (args.rvo or args.astar)
    if args.rvo or do_all:
        make_rvo_gif(os.path.join(OUT, "rvo_avoidance.gif"))
    if args.astar or do_all:
        make_astar_gif(os.path.join(OUT, "astar_search.gif"))


if __name__ == "__main__":
    main()
