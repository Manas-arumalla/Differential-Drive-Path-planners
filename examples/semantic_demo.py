"""Semantic navigation demo: terrain-aware A* vs plain geometric A*.

Shows the terrain cost map (road / grass / mud) with both paths overlaid: the
geometric planner cuts straight through expensive terrain, while the semantic
planner detours to stay on cheap terrain.

    python examples/semantic_demo.py   # -> media/semantic_nav.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navstack.environment import Environment
from navstack.planners.astar_planner import AStarPlanner
from navstack.planners.semantic_astar import SemanticAStarPlanner, SemanticCostmap

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")
plt.style.use("dark_background")


def main():
    env = Environment(30.0, 30.0, 0.3)
    cm = SemanticCostmap(env)
    cm.add_rect_cost(8, 8, 14, 9, 8.0)     # mud band (expensive)
    cm.add_rect_cost(8, 17, 14, 3, 2.5)    # grass strip (mildly expensive)
    start, goal = (3.0, 3.0), (27.0, 27.0)

    sem = SemanticAStarPlanner(env, cm, robot_radius=0.4).plan(start, goal)
    geo = AStarPlanner(env, robot_radius=0.4).plan(start, goal)

    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#1A202C")
    im = ax.imshow(cm.cost, origin="lower", extent=[0, env.width, 0, env.height],
                   cmap="inferno", alpha=0.85, vmin=1, vmax=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="traversal cost")
    if geo:
        g = np.array(geo); ax.plot(g[:, 0], g[:, 1], color="#FF6B6B", lw=3, label="geometric A* (through mud)")
    if sem:
        s = np.array(sem); ax.plot(s[:, 0], s[:, 1], color="#00FF88", lw=3, label="semantic A* (stays cheap)")
    ax.plot(*start, "o", color="#00FFFF", ms=13); ax.plot(*goal, "*", color="#FFFFFF", ms=20)
    ax.set_title("Semantic Navigation — terrain-aware planning", color="white", fontweight="bold")
    ax.legend(loc="lower right", facecolor="#2D3748", labelcolor="white")
    ax.set_aspect("equal")
    out = os.path.join(OUT, "semantic_nav.png")
    fig.savefig(out, dpi=150, facecolor="#1A202C", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
