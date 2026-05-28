"""Interactive navstack dashboard.

    pip install -e ".[dashboard]"
    streamlit run dashboard/streamlit_app.py

Pick a planner and start/goal, plan on the demo map, and inspect the path with
live metrics. Complements the desktop PySide6 control center for quick web demos.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from navstack.environment import create_demo_environment
from navstack.planners import (
    AStarPlanner, DijkstraPlanner, RRTPlanner, RRTStarPlanner,
    PRMPlanner, PSOPlanner, APFPlanner,
)

PLANNERS = {
    "A*": lambda env, r: AStarPlanner(env, robot_radius=r),
    "Dijkstra": lambda env, r: DijkstraPlanner(env, robot_radius=r),
    "RRT": lambda env, r: RRTPlanner(env, robot_radius=r, step_size=2.0, max_iterations=5000),
    "RRT*": lambda env, r: RRTStarPlanner(env, robot_radius=r, step_size=2.0, max_iterations=3000),
    "PRM": lambda env, r: PRMPlanner(env, robot_radius=r, num_samples=300, k_neighbors=10),
    "PSO": lambda env, r: PSOPlanner(env, robot_radius=r, num_particles=50, max_iterations=80),
    "APF": lambda env, r: APFPlanner(env, robot_radius=r, attractive_gain=5.0, repulsive_gain=150.0),
}


def path_length(path):
    return sum(np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
               for i in range(len(path) - 1)) if path and len(path) > 1 else 0.0


def main():
    st.set_page_config(page_title="navstack", layout="wide")
    st.title("navstack — Path Planner Explorer")

    env = create_demo_environment()
    with st.sidebar:
        name = st.selectbox("Planner", list(PLANNERS))
        radius = st.slider("Robot radius (m)", 0.2, 1.0, 0.5, 0.1)
        sx = st.slider("Start X", 1.0, 48.0, 3.0); sy = st.slider("Start Y", 1.0, 48.0, 3.0)
        gx = st.slider("Goal X", 1.0, 48.0, 45.0); gy = st.slider("Goal Y", 1.0, 48.0, 45.0)
        go = st.button("Plan")

    if go:
        planner = PLANNERS[name](env, radius)
        t0 = time.time()
        path = planner.plan((sx, sy), (gx, gy))
        dt = time.time() - t0

        fig, ax = plt.subplots(figsize=(8, 8), facecolor="#1A202C")
        ax.set_facecolor("#1A202C")
        ax.imshow(env.grid, origin="lower", extent=[0, env.width, 0, env.height], cmap="bone_r", alpha=0.85)
        ax.plot(sx, sy, "o", color="#00FF88", ms=12); ax.plot(gx, gy, "*", color="#FF4444", ms=20)
        if path:
            p = np.array(path); ax.plot(p[:, 0], p[:, 1], color="#00D4FF", lw=3)
        ax.set_aspect("equal"); ax.set_title(name, color="white")

        c1, c2 = st.columns([2, 1])
        c1.pyplot(fig)
        with c2:
            if path:
                st.metric("Path length (m)", f"{path_length(path):.2f}")
                st.metric("Waypoints", len(path))
                st.metric("Planning time (s)", f"{dt:.3f}")
            else:
                st.error("No path found.")


if __name__ == "__main__":
    main()
