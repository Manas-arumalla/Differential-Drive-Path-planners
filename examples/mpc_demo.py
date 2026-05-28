"""MPC path-following demo: track an S-curve with the LTV-MPC controller.

    python examples/mpc_demo.py   # -> media/mpc_follower.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navstack.controllers.mpc_follower import MPCPathFollower
from navstack.robot import DifferentialDriveRobot

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")
plt.style.use("dark_background")


def main():
    xs = np.linspace(0, 10, 50)
    path = [(float(x), float(2 * np.sin(0.5 * x))) for x in xs]
    mpc = MPCPathFollower(horizon=12, dt=0.1, v_ref=1.2, v_max=1.6, omega_max=2.5)
    robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0, max_speed=1.6, max_omega=2.5)

    traj = [(robot.x, robot.y)]
    for _ in range(400):
        v, w, _ = mpc.compute_control((robot.x, robot.y, robot.theta), path)
        robot.step(v, w, dt=0.1)
        traj.append((robot.x, robot.y))
        if np.hypot(robot.x - path[-1][0], robot.y - path[-1][1]) < 0.35:
            break
    traj = np.array(traj); ref = np.array(path)

    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#1A202C")
    ax.set_facecolor("#1A202C")
    ax.plot(ref[:, 0], ref[:, 1], "--", color="#FFA500", lw=2, label="reference path")
    ax.plot(traj[:, 0], traj[:, 1], color="#00D4FF", lw=3, label="MPC trajectory")
    ax.plot(0, 0, "o", color="#00FF88", ms=12, label="start")
    ax.plot(ref[-1, 0], ref[-1, 1], "*", color="#FF4444", ms=20, label="goal")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
    ax.legend(loc="upper right", facecolor="#2D3748", labelcolor="white")
    ax.set_title("MPC Path Following (LTV-MPC tracker)", color="white", fontweight="bold")
    out = os.path.join(OUT, "mpc_follower.png")
    fig.savefig(out, dpi=150, facecolor="#1A202C", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
