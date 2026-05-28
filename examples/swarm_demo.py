"""Multi-robot swarm navigation demo (reciprocal RVO circle swap).

    python examples/swarm_demo.py        # -> media/swarm_rvo.gif

Eight robots on a circle each drive to the antipodal point; every path crosses
the centre, so collision-free coordination requires reciprocal avoidance.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from navstack.controllers.swarm import simulate_circle_swap

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")
plt.style.use("dark_background")


def main(n_robots=8, fps=25):
    res = simulate_circle_swap(n_robots=n_robots, circle_radius=5.0, robot_radius=0.35, steps=600)
    trajs, goals = res["trajectories"], res["goals"]
    print(f"all_reached={res['all_reached']} min_clearance={res['min_clearance']:.3f} m")

    T = max(len(t) for t in trajs)
    trajs = [np.vstack([t, np.repeat(t[-1:], T - len(t), axis=0)]) if len(t) < T else t for t in trajs]
    colors = plt.cm.turbo(np.linspace(0, 1, n_robots))

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#1A202C")

    def draw(f):
        ax.clear(); ax.set_facecolor("#1A202C")
        for i in range(n_robots):
            tr = trajs[i][:f + 1]
            ax.plot(tr[:, 0], tr[:, 1], color=colors[i], lw=1.5, alpha=0.6)
            ax.add_patch(plt.Circle(tr[-1], 0.35, color=colors[i], alpha=0.9))
            ax.plot(goals[i, 0], goals[i, 1], "x", color=colors[i], ms=8, alpha=0.5)
        ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5); ax.set_aspect("equal")
        ax.grid(True, alpha=0.12)
        ax.set_title(f"Swarm Navigation — {n_robots} robots, reciprocal RVO", color="white", fontweight="bold")

    step = max(1, T // 160)  # cap frames
    frames = list(range(0, T, step))
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "swarm_rvo.gif")
    anim.save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"Saved {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
